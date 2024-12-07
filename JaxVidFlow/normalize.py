import functools
import math
from typing import Sequence

import jax
from jax import numpy as jnp

from JaxVidFlow import compat

@functools.partial(jax.jit, static_argnames=[
    'temporal_smoothing', 'max_gain', 'downsample_win'])
def normalize(img: jnp.ndarray, last_frame_mins: jnp.ndarray | None, last_frame_maxs: jnp.ndarray | None,
              temporal_smoothing: float = 0.05, max_gain: float = 10.0,
              downsample_win: int = 8, quantile_low: float = 0.0005,
              quantile_high: float=0.995, whitepoint=0.9, blackpoint=0.1) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Normalize an image with historgram stretching.

  Args:
    img: Image (HxWxC)
    last_frame_mins: Last frame per-channel minimums (C)
    last_frame_maxs: Last frame per-channel maximums (C)
    temporal_smoothing: Strength of temporal smoothing (0.0 = fixed gains from first frame, 1.0 = no smoothing)
    max_gain: Maximum per-channel gain. If the range would be stretched beyond this factor, the white point (top of range
      in output) is shifted down.
    downsample_win: Optional window for downsampling before computing channel max values. This smoothes out outliers
        from either very bright pixels in the scene or noise.

  Returns:
    normalized_image, mins, maxs
  """
  if downsample_win > 1:
    img_ds = compat.window_reduce_mean(img, (downsample_win, downsample_win))
  else:
    img_ds = img
  quantiles = jnp.quantile(img_ds, jnp.array([quantile_low, quantile_high], dtype=jnp.float32), axis=(0, 1))
  assert quantiles.shape == (2, 3)
  mins = quantiles[0]
  maxs = quantiles[1]
  # maxs = jnp.max(img_ds, axis=(0, 1))
  # mins = jnp.min(img_ds, axis=(0, 1))
  if last_frame_mins is not None and last_frame_maxs is not None:
    maxs = maxs * temporal_smoothing + last_frame_maxs * (1.0 - temporal_smoothing)
    mins = mins * temporal_smoothing + last_frame_mins * (1.0 - temporal_smoothing)
  ranges = maxs - mins
  gains = jnp.minimum((whitepoint - blackpoint) / ranges, max_gain)
  img -= mins
  img *= gains
  img += blackpoint
  return jnp.clip(img, 0.0, 1.0), mins, maxs