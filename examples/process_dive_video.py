import functools
import logging
import math
import os
import platform
import queue
import sys
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
import numpy as np
from tqdm import tqdm

sys.path.append('.')

from JaxVidFlow import colourspaces, lut, nlmeans, scale, utils
from JaxVidFlow.config import Config
from JaxVidFlow.video_reader import VideoReader
from JaxVidFlow.video_writer import VideoWriter


logger = logging.getLogger(__name__)
logging.basicConfig(encoding='utf-8', level=logging.INFO,
                    format='%(asctime)s.%(msecs)04d:%(filename)s:%(funcName)s:%(lineno)s:%(levelname)s: %(message)s',)


def normalize(x: jnp.ndarray, max_gain: float = 200.0, downsample_win: int = 8):
  # Ideally we want to use quantiles here to reduce the effect of noise, but quantiles are very slow
  # on GPU, so let's just do a downsample (reduce mean) instead.
  x_ds = jax.lax.reduce_window(x, init_value=0.0, computation=jax.lax.add,
                               window_dimensions=(downsample_win, downsample_win, 1),
                               window_strides=(downsample_win, downsample_win, 1), padding='valid')
  maxs = jnp.max(jnp.max(x_ds, axis=1), axis=0) / (downsample_win * downsample_win)
  mins = jnp.min(jnp.min(x_ds, axis=1), axis=0) / (downsample_win * downsample_win)
  ranges = maxs - mins
  gains = jnp.minimum(1.0 / ranges, max_gain)
  x = jnp.clip((x - mins), 0.0, 1.0) * gains
  return x


# Approximately equivalent to for benchmarking purposes:
# ffmpeg -i test_files/dji_dlogm_street.mp4 -vf "lut3d=file=luts/D_LOG_M_to_Rec_709_LUT_ZG_Rev1.cube:interp=trilinear,nlmeans=s=8.0:p=3:r=5,format=yuv420p" -c:v hevc_nvenc -f null -
@functools.partial(jax.jit, static_argnames=['frame_format', 'nlmeans_params'])
def process_frame(raw_frame, frame_format: str, nlmeans_params: frozenset[str, int]) -> jnp.ndarray:
  frame_in = VideoReader.DecodeFrame(raw_frame, frame_format)
  frame_in = scale.scale_image(frame_in, new_width=1920)
  frame = colourspaces.Rec709ToLinear(frame_in)
  #frame = nlmeans.nlmeans(frame, **dict(nlmeans_params))
  frame = normalize(frame)
  frame = colourspaces.LinearToRec709(frame)
  frame = utils.MergeSideBySide(frame_in, frame)
  return VideoWriter.EncodeFrame(frame)


def calculate_nlmeans_params(raw_frame, frame_format: str) -> frozenset[str, int]:
  frame_in = VideoReader.DecodeFrame(raw_frame, frame_format)
  frame = colourspaces.Rec709ToLinear(frame_in)
  print(f'First frame sigma: {np.array(utils.EstimateNoiseSigma(frame))}')
  return frozenset(nlmeans.default_nlmeans_params(frame).items())


def main():
  # Enable the persistent compilation cache so we are not recompiling every execution.
  utils.EnablePersistentCache()

  # Get default configs.
  config = Config(force_cpu_backend=False)

  if config.force_cpu_backend:
    jax.config.update('jax_platform_name', 'cpu')

  video_reader = VideoReader(filename='test_files/lionfish.mp4')

  codec_name, codec_options = utils.FindCodec(config.encoders)

  if config.profiling:
    jax.profiler.start_trace("/tmp/tensorboard")

  nlmeans_params = None

  with VideoWriter(filename='test_out.mp4',
                   frame_rate=video_reader.frame_rate(),
                   pixfmt='yuv420p',
                   codec_name=codec_name,
                   codec_options=codec_options) as video_writer:
    for frame_i, frame_data in tqdm(enumerate(video_reader), unit=' frames'):
      raw_frame, frame_format = frame_data

      if nlmeans_params is None:
        nlmeans_params = calculate_nlmeans_params(raw_frame, frame_format)
        print(f'NLmeans params: {nlmeans_params}')

      # Submit a processing call to the GPU.
      frame = process_frame(raw_frame, frame_format, nlmeans_params)

      video_writer.add_frame(encoded_frame=frame)
      video_writer.write_audio_packets(audio_packets=video_reader.audio_packets(),
                                       in_audio_stream=video_reader.audio_stream())
      video_reader.clear_audio_packets()

  if config.profiling:
    jax.profiler.stop_trace()


if __name__ == '__main__':
  main()
