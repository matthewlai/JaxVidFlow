import os
import platform
import sys

if platform.system() == 'Darwin':
  # Required for Jax on Metal (https://developer.apple.com/metal/jax/):
  os.environ['ENABLE_PJRT_COMPATIBILITY'] = '1'

import numpy as np

sys.path.append('.')

from JaxVidFlow import lut

def test_lookup_interpolate():
  l_in = np.array([[[0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.2]],
                   [[0.0, 0.2, 0.0],
                    [0.7, 0.0, 0.2]]])
  expected = np.array([[[0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.4]],
                       [[0.0, 0.4, 0.0],
                        [1.0, 0.0, 0.4]]])

  # This LUT should double everything below 0.5, then saturate at 1.0.
  out = lut.apply_lut(l_in, 'luts/double_lut_testing.cube')
  np.testing.assert_allclose(out, expected, atol=1e-4)