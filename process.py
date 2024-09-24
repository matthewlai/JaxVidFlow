import functools
import math
import os
import platform
import queue
import threading
import time
from typing import Any, Generator, Sequence

if platform.system() == 'Darwin':
  # Required for Jax on Metal (https://developer.apple.com/metal/jax/):
  os.environ['ENABLE_PJRT_COMPATIBILITY'] = '1'

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={os.cpu_count()}'

import av
import jax
from jax import numpy as jnp
from jax import scipy as jsp
import numpy as np
from tqdm import tqdm

from video_reader import VideoReader
from video_writer import VideoWriter


# Encoders listed by preference.
_ENCODERS = [
  # Test: ffmpeg -i test_files/dolphin_4096.mp4 -c:v {encoder_name} -f null -

  # Apple
  ('hevc_videotoolbox', None),
  
  # NVIDIA
  ('hevc_nvenc', None),

  # Software fallback
  ('hevc', None),
]


# Force x264 software H264 encoder at veryfast preset (for testing).
_FORCE_X264 = False
_FORCE_CPU = False
_PROFILING = False

_X264_OPTIONS = {
  'preset': 'superfast',
  'crf': '24'
}

# Force CPU for testing.
if _FORCE_CPU:
  jax.config.update('jax_platform_name', 'cpu')

@jax.jit
def normalize(rgb: jnp.ndarray) -> jnp.ndarray:
  max_r = jnp.max(rgb[:, :, 0]) + 0.001
  max_g = jnp.max(rgb[:, :, 1]) + 0.001
  max_b = jnp.max(rgb[:, :, 2]) + 0.001
  rgb = rgb.at[:, :, 0].multiply(1.0 / max_r)
  rgb = rgb.at[:, :, 1].multiply(1.0 / max_g)
  rgb = rgb.at[:, :, 2].multiply(1.0 / max_b)
  return rgb

@functools.partial(jax.jit, static_argnames=['frame_format'])
def process_frame(raw_frame, frame_format: str) -> jnp.ndarray:
  frame = VideoReader.DecodeFrame(raw_frame, frame_format)
  frame = normalize(frame)
  return VideoWriter.EncodeFrame(frame)


def main():
  video_reader = VideoReader(filename='test_files/dolphin_4096.mp4')

  codec_name = ''
  codec_options = None
  if _FORCE_X264:
    codec_name, codec_options = 'h264', _X264_OPTIONS
  else:
    for codec_name, codec_options in _ENCODERS:
      if VideoWriter.test_codec(codec_name=codec_name):
        break
  if not codec_name:
    raise RuntimeError(f'No valid codec found.')  

  start_time = time.time()

  first_frame = True

  if _PROFILING:
    jax.profiler.start_trace("/tmp/tensorboard")

  with VideoWriter(filename='test_out.mp4',
                   width=video_reader.width(),
                   height=video_reader.height(),
                   frame_rate=video_reader.frame_rate(),
                   pixfmt='yuv420p',
                   codec_name=codec_name,
                   codec_options=codec_options) as video_writer:
    for frame_i, frame_data in tqdm(enumerate(video_reader), unit=' frames'):
      raw_frame, frame_format = frame_data

      # Submit a processing call to the GPU.
      frame = process_frame(raw_frame, frame_format)

      video_writer.add_frame(encoded_frame=frame)

  duration = time.time() - start_time
  print(f'{frame_i} frames decoded, took {duration:.2f}s, FT: {duration/frame_i*1000:.2f} ms, {frame_i/duration:.3f} fps')

  if _PROFILING:
    jax.profiler.stop_trace()


if __name__ == '__main__':
  main()
