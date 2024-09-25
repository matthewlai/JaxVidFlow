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

import av
import jax
from jax import image as jax_image
from jax import numpy as jnp
from jax import scipy as jsp
import numpy as np
from tqdm import tqdm

from config import Config
import lut
import utils
from video_reader import VideoReader
from video_writer import VideoWriter


LUT_PATH='lut/D_LOG_M_to_Rec_709_LUT_ZG_Rev1.cube'


def normalize(x: jnp.ndarray):
  max_r, max_g, max_b = jnp.max(x[:, :, 0]), jnp.max(x[:, :, 1]), jnp.max(x[:, :, 2])
  x = x.at[:, :, 0].multiply(1.0 / max_r)
  x = x.at[:, :, 1].multiply(1.0 / max_g)
  x = x.at[:, :, 2].multiply(1.0 / max_b)
  return x


# ffmpeg -i test_files/dolphin_4096.mp4 -vf "lut3d=file=lut/D_LOG_M_to_Rec_709_LUT_ZG_Rev1.cube:interp=trilinear,normalize,format=yuv420p" -c:v hevc_nvenc -f null -
@functools.partial(jax.jit, static_argnames=['frame_format'])
def process_frame(raw_frame, frame_format: str) -> jnp.ndarray:
  frame = VideoReader.DecodeFrame(raw_frame, frame_format)
  frame = lut.apply_lut(frame, LUT_PATH)
  frame = normalize(frame)
  return VideoWriter.EncodeFrame(frame)


def main():
  # Get default configs.
  config = Config(force_cpu_backend=False)

  if config.force_cpu_backend:
    jax.config.update('jax_platform_name', 'cpu')

  video_reader = VideoReader(filename='test_files/dolphin_4096.mp4')

  codec_name, codec_options = utils.FindCodec(config.encoders)

  if config.profiling:
    jax.profiler.start_trace("/tmp/tensorboard")

  with VideoWriter(filename='test_out.mp4',
                   frame_rate=video_reader.frame_rate(),
                   pixfmt='yuv420p',
                   codec_name=codec_name,
                   codec_options=codec_options) as video_writer:
    for frame_i, frame_data in tqdm(enumerate(video_reader), unit=' frames'):
      raw_frame, frame_format = frame_data

      # Submit a processing call to the GPU.
      frame = process_frame(raw_frame, frame_format)

      video_writer.add_frame(encoded_frame=frame)

  if config.profiling:
    jax.profiler.stop_trace()


if __name__ == '__main__':
  main()
