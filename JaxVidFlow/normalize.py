import functools
import math
from typing import Sequence

import jax
from jax import numpy as jnp

from JaxVidFlow import compat

def _calculate_gains(channel_maxs: jnp.ndarray, last_frame_gains: jnp.ndarray, last_frame_gains_valid: jnp.ndarray,
                     strength: float, temporal_smoothing: float, max_gain: float):
  # First we calculate what the gain would be without temporal smoothing.
  current_frame_gains = jnp.minimum(1.0 / channel_maxs, max_gain)
  current_frame_gains = (current_frame_gains - 1.0) * strength + 1.0
  current_frame_gains = jnp.minimum(current_frame_gains,
                                    jnp.ones_like(current_frame_gains) * max_gain)
  
  # Then we apply temporal smoothing.
  last_frame_gains_weight = last_frame_gains_valid * (1.0 - temporal_smoothing)
  total_weight = temporal_smoothing + last_frame_gains_weight
  total_gains = current_frame_gains * temporal_smoothing + last_frame_gains * last_frame_gains_weight
  return total_gains / total_weight
  

@functools.partial(jax.jit, static_argnames=[
    'strength', 'temporal_smoothing', 'max_gain', 'downsample_win'])
def normalize(img: jnp.ndarray, last_frame_gains: jnp.ndarray, last_frame_gains_valid: jnp.ndarray,
              strength: float = 1.0, temporal_smoothing: float = 0.05, max_gain: float = 200.0,
              downsample_win: int = 8) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Normalize an image with historgram stretching.

  Args:
    img: Image (HxWxC)
    last_frame_gains: Last frame per-channel gains (C)
    last_frame_gains_valid: Should be 1.0 if last_frame_gains is valid, otherwise 0.0 (eg. for the first frame)
    strength: Strength of the normalization (0.0 = no change, 1.0 = all channels stretched to 1.0)
    temporal_smoothing: Strength of temporal smoothing (0.0 = fixed gains from first frame, 1.0 = no smoothing)
    max_gain: Maximum per-channel gain.
    downsample_win: Optional window for downsampling before computing channel max values. This smoothes out outliers
        from either very bright pixels in the scene or noise.
  """
  if downsample_win > 1:
    img_ds = compat.window_reduce_mean(img, (downsample_win, downsample_win))
  else:
    img_ds = img
  maxs = jnp.max(jnp.max(img_ds, axis=1), axis=0)
  gains = _calculate_gains(channel_maxs=maxs, last_frame_gains=last_frame_gains, last_frame_gains_valid=last_frame_gains_valid,
                           strength=strength, temporal_smoothing=temporal_smoothing, max_gain=max_gain)
  return jnp.clip(img * gains, 0.0, 1.0), gains