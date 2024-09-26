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
