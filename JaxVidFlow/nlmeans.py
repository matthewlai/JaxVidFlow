import functools

import jax
from jax import image as jax_image
from jax import numpy as jnp
import jax.scipy as jsp
import numpy as np

from JaxVidFlow import colourspaces, lut, utils
from JaxVidFlow.types import FT

def _patch_around(img: np.ndarray, x: int, y: int, p_div_2: int) -> np.ndarray:
  min_y = y - p_div_2
  max_y = y + p_div_2
  min_x = x - p_div_2
  max_x = x + p_div_2
  return img[min_y:max_y + 1, min_x:max_x + 1]

def _inclusive_range(start: int, end: int, step: int = 1):
  return range(start, end + 1, step)

def naive_nlmeans(img: np.ndarray, strength: float, search_range: int, patch_size: int) -> np.ndarray:
  """Simple naive non-compilable implementation of NL-means to use for testing more efficient implementations."""
  assert search_range % 2 == 1, 'search_range must be odd'
  assert patch_size % 2 == 1, 'patch_size must be odd'
  p_div_2 = (patch_size - 1) // 2
  s_div_2 = (search_range - 1) // 2
  true_w = img.shape[1]
  true_h = img.shape[0]
  out_img = np.zeros_like(img)
  padding = p_div_2 + s_div_2
  img = np.pad(img, [(padding, padding), (padding, padding), (0, 0)], mode='edge')
  w = img.shape[1]
  h = img.shape[0]
  exp_scaling = 1.0 / (strength ** 2)
  for y in range(padding, h - padding):
    for x in range(padding, w - padding):
      total_weight = 0.0
      max_weight = 0.0
      weighted_total_new_pixel = np.zeros((img.shape[2],), dtype=np.float32)
      this_patch = _patch_around(img, x, y, p_div_2)

      for dx in _inclusive_range(-s_div_2, s_div_2):
        for dy in _inclusive_range(-s_div_2, s_div_2):
          if dx == 0 and dy == 0:
            continue
          candidate_x, candidate_y = x + dx, y + dy
          candidate_patch = _patch_around(img, candidate_x, candidate_y, p_div_2)
          patch_diff = candidate_patch - this_patch
          patch_diff_sq = patch_diff ** 2
          weight = np.mean(np.exp(-patch_diff_sq * exp_scaling))
          weighted_total_new_pixel += img[candidate_y, candidate_x] * weight
          total_weight += weight
          max_weight = max(max_weight, float(weight))

      # Finally, add the original pixel at the highest weight we have seen.
      weighted_total_new_pixel += img[y, x] * max_weight
      total_weight += max_weight
      inv_total_weight = 1.0 / total_weight
      out_img[y - padding, x - padding] = weighted_total_new_pixel * inv_total_weight
  return out_img

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

  exp_scaling = 1.0 / ((strength ** 2) * search_range * search_range)  # C in the paper, except we use lambda = strength ** 2

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
    patch_diffs = _separable_2d_conv(diff_sq, jnp.ones((patch_size, 1), dtype=FT()),
                                              jnp.ones((1, patch_size), dtype=FT()))  # This is v_n in the paper.
    weights = jnp.exp(-patch_diffs * exp_scaling)

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


@jax.jit
def psnr(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
  mse = jnp.max(jnp.array([jnp.mean((a - b) ** 2), jnp.asarray(0.0001)]))
  return 20 * jnp.log10(1.0 / jnp.sqrt(mse))