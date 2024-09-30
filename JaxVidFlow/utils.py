import math
from pathlib import Path
from typing import Sequence

import jax
from jax import numpy as jnp
from jax import scipy as jsp
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

def CompareTwo(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
  shape = arr1.shape
  assert arr1.shape == arr2.shape
  out = arr1.copy()
  center = shape[1] // 2
  out[:, center:] = arr2[:, center:]
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
  ], dtype=FT())

  plane_sigmas = []

  for i in range(img.shape[2]):
    sigma = jnp.abs(jsp.signal.convolve2d(img[:, :, i], kernel, mode='same'))
    sigma = jnp.sum(sigma)
    sigma *= jnp.sqrt(0.5 * math.pi) / (6.0 * (w - 2) * (h - 2))
    plane_sigmas.append(sigma)
  return jnp.stack(plane_sigmas)

