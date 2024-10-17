import os
import platform
import sys

import jax
from jax import numpy as jnp
import numpy as np

sys.path.append('.')

from JaxVidFlow import compat

def test_window_reduce_sum():
  img = jnp.array([
      0.0, 1.0, 2.0, 3.0,
      1.0, 2.0, 3.0, 4.0,
      2.0, 3.0, 4.0, 5.0,
      3.0, 4.0, 5.0, 6.0
  ])
  img = jnp.reshape(img, (4, 4, 1))
  reduced = compat.window_reduce_mean(img, (2, 2))
  expected = jnp.array([
      1.0, 3.0,
      3.0, 5.0,
  ])
  expected = jnp.reshape(expected, (2, 2, 1))
  np.testing.assert_allclose(reduced, expected, atol=1e-4)