import os
import platform
import sys

import numpy as np

sys.path.append('.')

from JaxVidFlow import colourspaces

def test_LinearToRec709():
  l_in = np.array([0.01, 0.5, 1.0])
  expected = np.array([0.045, 0.705515, 1.0])
  out = colourspaces.LinearToRec709(l_in)
  np.testing.assert_allclose(out, expected, atol=1e-4)

def test_LinearToRec709_roundtrip():
  l_in = np.array([0.01, 0.5, 1.0])
  out = colourspaces.Rec709ToLinear(colourspaces.LinearToRec709(l_in))
  np.testing.assert_allclose(out, l_in, atol=1e-4)