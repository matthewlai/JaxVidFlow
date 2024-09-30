import functools
import math

import jax
from jax import image as jax_image
from jax import numpy as jnp
import jax.scipy as jsp
import numpy as np

from JaxVidFlow import colourspaces, lut, utils
from JaxVidFlow.types import FT


def _make_offsets(search_range: int) -> np.ndarray:
  offsets = []
  s_div_2 = (search_range - 1) // 2
  for nx in _inclusive_range(-s_div_2, s_div_2):
    for ny in _inclusive_range(-s_div_2, 0):
      if search_range * ny + nx < 0:
        offsets.append(np.array([ny, nx]))
  return np.stack(offsets)

@functools.partial(jax.jit, static_argnames=['strength', 'search_range', 'patch_size'])
def nlmeans(img: jnp.ndarray, strength: float, search_range: int, patch_size: int) -> jnp.ndarray:
  """Fast convolution-based NL-means.

  This implements NL-means using convolutions by swapping the outer 2 loops. This is described in
  "A Simple Trick to Speed Up and Improve the Non-Local Means" by Laurent Condat.
  """

  # Note that ffmpeg and the paper use different definition of patch size and search range. We are using
  # ffmpeg's definition of (search_range * search_range) window instead of (2*search_range+1, 2*search_range+1) windows.
  assert search_range % 2 == 1, 'search_range must be odd'
  assert patch_size % 2 == 1, 'patch_size must be odd'

  p_div_2 = (patch_size - 1) // 2
  s_div_2 = (search_range - 1) // 2
  padding = p_div_2 + s_div_2
  offsets = jnp.array(_make_offsets(search_range))

  def _pad(a: jnp.ndarray) -> jnp.ndarray:
    return jnp.pad(a, [(padding, padding), (padding, padding), (0, 0)], mode='edge')

  def _separable_2d_conv(img: jnp.ndarray, kernel_0: jnp.ndarray, kernel_1: jnp.ndarray) -> jnp.ndarray:
    output_in_dims = []
    # Doing the channels separately seems to result in much less VRAM needed for intermediates.
    for dim in range(3):
      img_slice = img[:, :, dim]
      x = jsp.signal.convolve(img_slice, kernel_0, mode='same')
      x = jsp.signal.convolve(x, kernel_1, mode='same')
      output_in_dims.append(x)
    return jnp.stack(output_in_dims, axis=2)

  padded_img = _pad(img)

  def _process_offset_body_fn(i, carry) -> jnp.ndarray:
    img_out, weight_sum, weight_max = carry
    offset_y = offsets[i][0]
    offset_x = offsets[i][1]
    offset_img = jax.lax.dynamic_slice(padded_img, (offset_y + padding, offset_x + padding, 0), img.shape)
    diff_sq = (img - offset_img) ** 2  # This is u_n in the paper.

    # We can in theory do this faster by updating the values incrementally since we have a constant kernel. However,
    # that sounds difficult to implement efficiently on GPU, and these separable 2D convs are already very fast.
    patch_diffs = _separable_2d_conv(diff_sq, jnp.ones((patch_size, 1), dtype=FT()) / patch_size,
                                              jnp.ones((1, patch_size), dtype=FT()) / patch_size)  # This is v_n in the paper.
    weights = jnp.exp(-patch_diffs / (strength ** 2))

    # Line 5) in the pseudo-code.
    img_out = img_out + weights * offset_img
    weight_sum += weights
    weight_max = jnp.maximum(weight_max, weights)

    # Line 6) in the pseudo-code required to support only doing half of the offsets and relying on symmetry.
    opposite_offset_img = jax.lax.dynamic_slice(padded_img, (-offset_y + padding, -offset_x + padding, 0), img.shape)
    padded_weights = _pad(weights)
    opposite_offset_weights = jax.lax.dynamic_slice(padded_weights, (-offset_y + padding, -offset_x + padding, 0), img.shape)
    img_out = img_out + opposite_offset_weights * opposite_offset_img
    weight_sum += opposite_offset_weights
    weight_max = jnp.maximum(weight_max, opposite_offset_weights)

    return (img_out, weight_sum, weight_max)

  init_val = (jnp.zeros_like(img), jnp.zeros_like(img), jnp.zeros_like(img))
  img_out, weight_sum, weight_max = jax.lax.fori_loop(
    lower=0, upper=offsets.shape[0], body_fun=_process_offset_body_fn, init_val=init_val)

  # Add the contribution of the original pixels.
  img_out = img_out + img * weight_max

  # Normalise.
  img_out = img_out / (weight_sum + weight_max)

  return img_out


@functools.partial(jax.jit, static_argnames=['search_range', 'patch_size'])
def nlmeans_patchwise(img: jnp.ndarray, search_range: int, patch_size: int,
                      strength: jnp.ndarray | None = None, sigma: jnp.ndarray | None = None) -> jnp.ndarray:
  """Patchwise Non-Local Means as described in https://www.ipol.im/pub/art/2011/bcm_nlm/revisions/2022-01-01/article.pdf.

  Args:
    img: Image.
    strength: h parameter in the paper. Can be a scalar or per-channel. None to automatically calculate from sigma.
    sigma: Noise estimation sigma. Estimated automatically if None.
    search_range: Search range.
    patch_size: Patch size.

  Returns:
    Denoised image.
  """
  assert search_range % 2 == 1, 'search_range must be odd'
  assert patch_size % 2 == 1, 'patch_size must be odd'

  if sigma is None:
    sigma = utils.EstimateNoiseSigma(img)

  if strength is None:
    if sigma is None:
      raise ValueError(f'At least one of strength and sigma must be specified.')
    if patch_size == 3:
      strength = 0.55 * sigma
    elif patch_size == 5:
      strength = 0.4 * sigma
    elif patch_size == 7:
      strength = 0.35 * sigma
    else:
      raise ValueError(f'strength must be specified with patch size {patch_size}')

  if sigma is None:
    sigma = jnp.zeros_like(strength)

  p_div_2 = (patch_size - 1) // 2
  s_div_2 = (search_range - 1) // 2

  num_patches = int(math.ceil(img.shape[0] / patch_size)), int(math.ceil(img.shape[1] / patch_size))

  # First we pad the image. Since we are aligning blocks at (0, 0), on the sides adjacent to that corner, we only need to pad s_div_2.
  # However, on the (1, 1) sides we need s_div_2 + padding we need to get width/height to a multiple of patch size.
  pad_low_side = s_div_2
  pad_high_side = (s_div_2 + num_patches[0] * patch_size - img.shape[0]), (s_div_2 + num_patches[1] * patch_size - img.shape[1])
  padded_img = jnp.pad(img, ((pad_low_side, pad_high_side[0]), (pad_low_side, pad_high_side[1]), (0, 0)))

  img_out = jnp.zeros_like(padded_img)

  dist_baseline = 2.0 * sigma ** 2
  h_sq = strength ** 2

  search_offsets = []
  for y in range(-s_div_2, s_div_2 + 1):
    for x in range(-s_div_2, s_div_2 + 1):
      search_offsets.append(np.array((y, x), dtype=np.int32))
  search_offsets = jnp.stack(search_offsets)

  ref_img = jax.lax.dynamic_slice(padded_img, (pad_low_side, pad_low_side, 0),
                                  (num_patches[0] * patch_size, num_patches[1] * patch_size, 3))

  def _repeat_weights(w) -> jnp.ndarray:
    x = jnp.repeat(w, repeats=patch_size, axis=0, total_repeat_length=num_patches[0] * patch_size)
    return jnp.repeat(x, repeats=patch_size, axis=1, total_repeat_length=num_patches[1] * patch_size)[:, :, jnp.newaxis]

  def _process_offset(i, carry) -> jnp.ndarray:
    out_img, total_weights, max_weights = carry
    offset_img = jax.lax.dynamic_slice(padded_img, (pad_low_side + search_offsets[i][0], pad_low_side + search_offsets[i][1], 0),
                                       (num_patches[0] * patch_size, num_patches[1] * patch_size, 3))
    dist = (offset_img - ref_img) ** 2

    # Here we depart from the paper a bit - instead of averaging across channels, we keep a per-channel distance for each patch. This allows us
    # to use channel-dependent sigma baselines and h.
    patch_sums = jax.lax.reduce_window(dist, init_value=0.0, computation=jax.lax.add,
                                       window_dimensions=(patch_size, patch_size, 1), window_strides=(patch_size, patch_size, 1), padding='valid')
    assert patch_sums.shape == num_patches + (3,)
    d_sq = patch_sums / (patch_size ** 2)

    # Now we can compute per-channel max(d^2 - 2sigma^2, 0) / h^2, then take the mean over channels.
    exp = -jnp.mean(jnp.maximum(d_sq - 2.0 * sigma ** 2, 0.0) / (strength ** 2), axis=2)
    weights = jnp.exp(exp)

    total_weights += weights
    max_weights = jnp.maximum(max_weights, weights)
    out_img += offset_img * _repeat_weights(weights)
    return out_img, total_weights, max_weights

  # For the output image we only need to pad to multiple of patch size.
  out_img = jnp.zeros((num_patches[0] * patch_size, num_patches[1] * patch_size, 3), dtype=img.dtype)
  total_weights = jnp.zeros(num_patches, dtype=img.dtype)
  max_weights = jnp.zeros_like(total_weights)
  init = out_img, total_weights, max_weights
  out_img, total_weights, max_weights = jax.lax.fori_loop(lower=0, upper=search_offsets.shape[0], body_fun=_process_offset, init_val=init, unroll=16)
  
  # Add contributions from the reference patches.
  out_img += ref_img * _repeat_weights(max_weights)

  total_weights += max_weights

  # Normalize.
  out_img /= _repeat_weights(total_weights)

  return jax.lax.dynamic_slice(out_img, (0, 0, 0), img.shape)
