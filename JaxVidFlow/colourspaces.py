import jax
from jax import numpy as jnp
import numpy as np

def LinearToRec709(x: jnp.ndarray) -> jnp.ndarray:
  # Rec709 is 4.5x for x < 0.018, and 1.099x^0.45 - 0.099 otherwise.
  mask_lt_threshold = x < 0.018
  return 4.5 * x * mask_lt_threshold + (1.099 * jnp.power(x, 0.45) - 0.099) * (1.0 - mask_lt_threshold)

def Rec709ToLinear(x: jnp.ndarray) -> jnp.ndarray:
  mask_lt_threshold = x < 0.081
  return (1.0 / 4.5) * x * mask_lt_threshold + jnp.power(((x + 0.099) / 1.099), 1.0 / 0.45) * (1.0 - mask_lt_threshold)

def YUV2RGB(x: jnp.ndarray) -> jnp.ndarray:
  y = x[:, :, 0]
  u = x[:, :, 1] - 0.5
  v = x[:, :, 2] - 0.5
  r = y + 1.5748 * v
  g = y - 0.1873 * u - 0.4681 * v
  b = y + 1.8556 * u
  return jnp.clip(jnp.stack((r, g, b), axis=2), min=0.0, max=1.0)

def RGB2YUV(x: jnp.ndarray) -> jnp.ndarray:
  r, g, b = x[:, :, 0], x[:, :, 1], x[:, :, 2]
  y = 0.2126 * r + 0.7152 * g + 0.0722 * b
  u = -0.1146 * r + -0.3854 * g + 0.5 * b
  v = 0.5 * r + -0.4542 * g + -0.0458 * b
  u += 0.5
  v += 0.5
  return jnp.clip(jnp.stack((y, u, v), axis=2), min=0.0, max=1.0)