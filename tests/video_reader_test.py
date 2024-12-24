import fractions
import os
import platform
import sys
import time

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
  reader.seek(1.5)
  frame = next(reader)
  assert frame.frame_time == pytest.approx(1.5, 0.1)
  reader.seek(6.7)
  frame = next(reader)
  assert frame.frame_time == pytest.approx(6.7, 0.1)
  reader.seek(20.0)
  frame = next(reader)
  assert frame.frame_time == pytest.approx(8.08, 0.1)

def test_video_reader():
  reader = video_reader.VideoReader('test_files/lionfish.mp4')
  last_time = None
  for i, frame in enumerate(reader):
    expected_time = i * (1.0 / 25.0)
    assert frame.frame_time == pytest.approx(expected_time, abs=0.05)
    assert last_time is None or frame.frame_time >= last_time
    last_time = frame.frame_time
  assert last_time == pytest.approx(reader.duration(), abs=0.05)
