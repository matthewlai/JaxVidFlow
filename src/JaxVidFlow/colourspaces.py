import jax
from jax import numpy as jnp
import numpy as np

# Reference black is 16 in 8-bit, and 64 in 10-bit.
# Reference white is 235 in 8-bit, and 940 in 10-bit.
# They map to the same ratios.
_REC709_LOW = 16.0 / 256.0
_REC709_HIGH = 235.0 / 256.0

def LinearToRec709(x: jnp.ndarray, limited_range: bool = False) -> jnp.ndarray:
  # Rec709 is 4.5x for x < 0.018, and 1.099x^0.45 - 0.099 otherwise.
  mask_lt_threshold = x < 0.018
  full_range = 4.5 * x * mask_lt_threshold + (1.099 * jnp.power(x, 0.45) - 0.099) * (1.0 - mask_lt_threshold)
  if not limited_range:
    return full_range
  full_range = jnp.clip(full_range, 0.0, 1.0)
  return _REC709_LOW + full_range * (_REC709_HIGH - _REC709_LOW)

def Rec709ToLinear(x: jnp.ndarray, limited_range: bool = False) -> jnp.ndarray:
  mask_lt_threshold = x < 0.081
  x = (1.0 / 4.5) * x * mask_lt_threshold + jnp.power(((x + 0.099) / 1.099), 1.0 / 0.45) * (1.0 - mask_lt_threshold)
  if not limited_range:
    return x
  limited = jnp.clip(x, _REC709_LOW, _REC709_HIGH)
  return (limited - _REC709_LOW) / (_REC709_HIGH - _REC709_LOW)

def YUV2RGB(x: jnp.ndarray) -> jnp.ndarray:
  y = x[:, :, 0]
  u = x[:, :, 1] - 0.5
  v = x[:, :, 2] - 0.5

  # This is much faster than matrix multiply on CPU. On GPU it's the same.
  r = y + 1.5748 * v
  g = y - 0.1873 * u - 0.4681 * v
  b = y + 1.8556 * u
  rgb = jnp.stack((r, g, b), axis=2)
  return jnp.clip(rgb, min=0.0, max=1.0)

def RGB2YUV(x: jnp.ndarray) -> jnp.ndarray:
  r, g, b = x[:, :, 0], x[:, :, 1], x[:, :, 2]
  matrix = jnp.transpose(jnp.array([
    [0.2126, 0.7152, 0.0722],
    [-0.1146, -0.3854, 0.5],
    [0.5, -0.4542, -0.0458]
  ], dtype=x.dtype))
  yuv = jnp.matmul(x, matrix)
  yuv = yuv.at[:, :, 1:3].add(0.5)
  return jnp.clip(yuv, min=0.0, max=1.0)