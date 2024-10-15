import os
import platform
import sys

from jax import numpy as jnp

if platform.system() == 'Darwin':
  # Required for Jax on Metal (https://developer.apple.com/metal/jax/):
  os.environ['ENABLE_PJRT_COMPATIBILITY'] = '1'

sys.path.append('.')

from JaxVidFlow import gyroflow

def test_gyroflow():
  gf = gyroflow.Gyroflow(gyroflow_lib_path='gyroflow/libgyroflow_frei0r.dylib',
                         gyroflow_project_path='test_files/dji_dlogm_street.gyroflow')
  in_frame = jnp.zeros((1080, 1920, 4), jnp.uint8)
  processed = gf.process_frame(frame=in_frame, frame_time=0.0)
  assert processed.shape == (1080, 1920, 4)
  assert processed.dtype == jnp.uint8