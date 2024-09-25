import jax
from jax import numpy as jnp

key = jax.random.key(42)
x = jax.random.uniform(key, (3, 3), minval=-5, maxval=5)
print(jnp.floor(x))