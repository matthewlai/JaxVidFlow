from pathlib import Path
from typing import Sequence

import jax
import numpy as np
from PIL import Image

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
