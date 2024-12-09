import fractions
import os
import platform
import sys

import jax
from jax import numpy as jnp
import numpy as np
import pytest

sys.path.append('.')

from JaxVidFlow import video_reader

def test_video_reader():
  reader = video_reader.VideoReader('test_files/lionfish.mp4')
  assert reader.width() == 5120
  assert reader.height() == 3840
  assert reader.frame_rate() == fractions.Fraction(25, 1)
  assert reader.num_frames() == 203
  assert reader.duration() == pytest.approx(8.12, 0.1)
  reader.seek(4.3)
  frame = next(reader)
  assert frame.frame_time == pytest.approx(4.3, 0.1)
  reader.seek(6.1)
  frame = next(reader)
  assert frame.frame_time == pytest.approx(6.1, 0.1)
  reader.seek(0.5)
  frame = next(reader)
  assert frame.frame_time == pytest.approx(0.5, 0.1)