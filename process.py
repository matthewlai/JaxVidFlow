import functools
import logging
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

import colourspaces
from config import Config
import lut
import utils
from video_reader import VideoReader
from video_writer import VideoWriter


logger = logging.getLogger(__name__)
logging.basicConfig(encoding='utf-8', level=logging.INFO,
                    format='%(asctime)s.%(msecs)04d:%(filename)s:%(funcName)s:%(lineno)s:%(levelname)s: %(message)s',)


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
  frame_in = VideoReader.DecodeFrame(raw_frame, frame_format)
  frame = frame_in
  frame = lut.apply_lut(frame, LUT_PATH)
  frame = colourspaces.Rec709ToLinear(frame)
  frame = normalize(frame)
  frame = colourspaces.LinearToRec709(frame)
  half_frame_width = frame.shape[1] // 2
  frame = frame.at[:, 0:half_frame_width].set(frame_in[:, 0:half_frame_width])
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
      video_writer.write_audio_packets(audio_packets=video_reader.audio_packets(),
                                       in_audio_stream=video_reader.audio_stream())
      video_reader.clear_audio_packets()

  if config.profiling:
    jax.profiler.stop_trace()


if __name__ == '__main__':
  main()
