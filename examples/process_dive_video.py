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

import psutil

if platform.system() == 'Darwin':
  # Required for Jax on Metal (https://developer.apple.com/metal/jax/):
  os.environ['ENABLE_PJRT_COMPATIBILITY'] = '1'

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

xla_flags = {
  'xla_force_host_platform_device_count': f'{psutil.cpu_count(logical=False)}',
}

if xla_flags:
  os.environ['XLA_FLAGS'] = '--' + ' '.join([f'{name}={val}' for name, val in xla_flags.items()])

import av
import jax
from jax import image as jax_image
from jax import numpy as jnp
import numpy as np
from tqdm import tqdm

sys.path.append('.')

from JaxVidFlow import colourspaces, compat, lut, nlmeans, normalize, scale, utils
from JaxVidFlow.config import Config
from JaxVidFlow.types import FT
from JaxVidFlow.video_reader import VideoReader
from JaxVidFlow.video_writer import VideoWriter


logger = logging.getLogger(__name__)
logging.basicConfig(encoding='utf-8', level=logging.INFO,
                    format='%(asctime)s.%(msecs)04d:%(filename)s:%(funcName)s:%(lineno)s:%(levelname)s: %(message)s',)

# Approximately equivalent to for benchmarking purposes:
# ffmpeg -i test_files/lionfish.mp4 -vf "scale=1920:-1,normalize,format=yuv420p" -c:v hevc_videotoolbox -f null -
@functools.partial(jax.jit, static_argnames=[])
def process_frame(frame_in, last_frame_gains, last_frame_gains_valid) -> tuple[jnp.ndarray, jnp.ndarray]:
  frame = colourspaces.Rec709ToLinear(frame_in)
  frame, last_frame_gains = normalize.normalize(
      img=frame, last_frame_gains=last_frame_gains, last_frame_gains_valid=last_frame_gains_valid)
  frame = colourspaces.LinearToRec709(frame)
  frame = utils.MergeSideBySide(frame_in, frame)
  return frame, last_frame_gains


def calculate_nlmeans_params(raw_frame, frame_format: str) -> frozenset[str, int]:
  frame_in = VideoReader.DecodeFrame(raw_frame, frame_format)
  frame = colourspaces.Rec709ToLinear(frame_in)
  print(f'First frame sigma: {np.array(utils.EstimateNoiseSigma(frame))}')
  return frozenset(nlmeans.default_nlmeans_params(frame).items())


def main():
  # Enable the persistent compilation cache so we are not recompiling every execution.
  utils.EnablePersistentCache()

  # Get default configs.
  config = Config(force_cpu_backend=False, profiling=False)

  if config.force_cpu_backend:
    jax.config.update('jax_platform_name', 'cpu')

  video_reader = VideoReader(filename='test_files/lionfish.mp4',
                             scale_width=1280)

  codec_name, codec_options = utils.FindCodec(config.encoders)

  if config.profiling:
    jax.profiler.start_trace("/tmp/jax-trace", create_perfetto_link=True)

  sharding = None
  
  last_frame_gains = jnp.zeros(3, dtype=FT())

  with VideoWriter(filename='test_out.mp4',
                   frame_rate=video_reader.frame_rate(),
                   pixfmt='yuv420p',
                   codec_name=codec_name,
                   codec_options=codec_options) as video_writer:
    for frame_i, frame_data in tqdm(enumerate(video_reader), unit=' frames'):
      if sharding is None:
        sharding = utils.GetSharding(num_devices_limit=None, divisor_of=frame_data.shape[0])

      frame_data = jax.device_put(frame_data, sharding)

      # Submit a processing call to the GPU.
      last_frame_gains_valid = jnp.ones(3, dtype=FT()) if frame_i > 0 else jnp.zeros(3, dtype=FT())
      frame, last_frame_gains = process_frame(
        frame_in=frame_data, last_frame_gains=last_frame_gains,
        last_frame_gains_valid=last_frame_gains_valid)

      video_writer.add_frame(frame=frame)
      video_writer.write_audio_packets(audio_packets=video_reader.audio_packets(),
                                       in_audio_stream=video_reader.audio_stream())
      video_reader.clear_audio_packets()

  if config.profiling:
    jax.profiler.stop_trace()


if __name__ == '__main__':
  main()
