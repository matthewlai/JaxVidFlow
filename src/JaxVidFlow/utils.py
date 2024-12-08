import functools
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import jax
from jax import numpy as jnp
from jax import scipy as jsp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np
from PIL import Image

from JaxVidFlow.types import FT
from JaxVidFlow.video_writer import VideoWriter

def FindCodec(candidates: Sequence[tuple[str, dict[str, str]]]) -> tuple[str, dict[str, str]]:
  codec_name = ''
  codec_options = None
  for codec_name, codec_options in candidates:
    if VideoWriter.test_codec(codec_name=codec_name):
      return codec_name, codec_options
  if not codec_name:
    raise RuntimeError(f'No valid codec found.')

def LoadImage(path: str) -> np.ndarray:
  with Image.open(path) as img:
    return np.array(img).astype(np.float32) / 255

def SaveImage(arr: np.ndarray, path: str) -> None:
  im = Image.fromarray((np.array(arr) * 255.0).astype(np.uint8))
  im.save(path)

def EnablePersistentCache(path: str | None = None) -> None:
  """Enables Jax persistent compilation cache."""
  if path is None:
    home = Path.home()
    path = str(home / '.jaxvidflow_jit_cache')
  jax.config.update('jax_compilation_cache_dir', path)
  jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
  jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.3)

def MergeSideBySide(arr1: jnp.ndarray, arr2: jnp.ndarray, axis=1) -> jnp.ndarray:
  arr1 = jnp.asarray(arr1)
  arr2 = jnp.asarray(arr2)
  shape = arr1.shape
  assert arr1.shape == arr2.shape
  out = arr1.copy()
  center = shape[axis] // 2
  if axis == 1:
    out = out.at[:, center:].set(arr2[:, center:])
  elif axis == 0:
    out = out.at[center:, :].set(arr2[center:, :])
  return out

@jax.jit
def PSNR(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
  mse = jnp.max(jnp.array([jnp.mean((a - b) ** 2), jnp.asarray(0.0001)]))
  return 20 * jnp.log10(1.0 / jnp.sqrt(mse))

@jax.jit
def EstimateNoiseSigma(img: jnp.ndarray) -> jnp.ndarray:
  """Estimate noise sigma based on 'Fast Noise Variance Estimation'

  J. Immerkær, "Fast Noise Variance Estimation", Computer Vision and Image Understanding, Vol. 64, No. 2, pp. 300-302, Sep. 1996

  """
  h, w = img.shape[:2]
  kernel = jnp.array([
    [1, -2, 1],
    [-2, 4, -2],
    [1, -2, 1]
  ], dtype=jnp.float32)  # Force the convolution to happen in F32 for stability.

  plane_sigmas = []

  for i in range(img.shape[2]):
    sigma = jnp.abs(jsp.signal.convolve2d(img[:, :, i], kernel, mode='same'))
    sigma = jnp.sum(sigma)
    sigma *= jnp.sqrt(0.5 * math.pi) / (6.0 * (w - 2) * (h - 2))
    plane_sigmas.append(sigma)
  return jnp.stack(plane_sigmas)

@functools.partial(jax.jit, static_argnames=['multiple_of'])
def PadFirstAxisToMultipleOf(x: jnp.ndarray, multiple_of: int) -> jnp.ndarray:
  old_shape0 = x.shape[0]
  new_shape0 = math.ceil(old_shape0 / multiple_of) * multiple_of
  padding = new_shape0 - old_shape0
  if padding == 0:
    return x
  else:
    return jnp.pad(x, ((0, padding), (0, 0) * (len(x.shape) - 1)))

def ExecuteWithPmap(fn: Callable[..., Any], args_to_pmap: Sequence[str],
                    output_leading_axes: Sequence[int] | int, **kwargs):
  """Execute fn by pmapping over the leading axis of args in args_to_pmap, and reconstruct afterwards."""
  num_devices = len(jax.devices())
  pm_fn = jax.pmap(fn, axis_name='slices')
  original_arg_shapes = {name: kwargs[name].shape for name in args_to_pmap}
  padded_args = {name: PadFirstAxisToMultipleOf(kwargs[name], multiple_of=num_devices)
                 for name in args_to_pmap}
  kwargs.update(padded_args)
  ret_vals = pm_fn(**kwargs)
  if isinstance(output_leading_axes, int):
    return ret_vals[:output_leading_axes]
  else:
    assert len(ret_vals) == len(output_leading_axes)
    return tuple(ret_vals[i][:output_leading_axes[i]] for i, size in enumerate(output_leading_axes))

@functools.cache
def GetSharding(num_devices_limit: int = None, divisor_of: int = 1):
  devices = jax.devices()
  num_devices = len(devices)
  if num_devices_limit is not None and num_devices > num_devices_limit:
    num_devices = num_devices_limit
  while divisor_of % num_devices != 0:
    num_devices -= 1
  devices = devices[:num_devices]
  print(f'Sharding over {len(devices)} devices')
  mesh = Mesh(devices=devices, axis_names=('s',))
  return NamedSharding(mesh, P('s'))