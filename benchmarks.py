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

from JaxVidFlow import colourspaces, lut, nlmeans, utils
from JaxVidFlow.types import FT
from JaxVidFlow.config import Config
from JaxVidFlow.video_reader import VideoReader
from JaxVidFlow.video_writer import VideoWriter


logger = logging.getLogger(__name__)
logging.basicConfig(encoding='utf-8', level=logging.WARN,
                    format='%(asctime)s.%(msecs)04d:%(filename)s:%(funcName)s:%(lineno)s:%(levelname)s: %(message)s',)


_LUT_PATH='luts/D_LOG_M_to_Rec_709_LUT_ZG_Rev1.cube'

_CODEC_PREFERENCES = [
    # Apple
    ('hevc_videotoolbox', None),
    
    # NVIDIA
    ('hevc_nvenc', None),

    # Software fallback
    ('hevc', None),
]

def _video_decode():
  start_time = time.time()
  for i, _ in enumerate(VideoReader(filename='test_files/dolphin_4096.mp4')):
    if i > 100:
      break
  duration = time.time() - start_time
  print(f' {i} frames in {duration:.2f}s ({i / duration:.2f} fps, {duration / i * 1000:.2f}ms/frame)')

def _video_transcode():
  start_time = time.time()
  video_reader = VideoReader(filename='test_files/dolphin_4096.mp4')

  codec_name, codec_options = utils.FindCodec(_CODEC_PREFERENCES)
  print(f' Using {codec_name}')
  with jax.default_device(jax.devices('cpu')[0]):
    with VideoWriter(filename='test_out.mp4',
                     frame_rate=video_reader.frame_rate(),
                     pixfmt='yuv420p',
                     codec_name=codec_name,
                     codec_options=codec_options) as video_writer:
      for i, frame_data in enumerate(video_reader):
        raw_frame, frame_format = frame_data
        y, u, v = raw_frame
        stacked_frame = jnp.concatenate([y.reshape(-1), u.reshape(-1), v.reshape(-1)]).reshape(-1, y.shape[1])
        video_writer.add_frame(encoded_frame=stacked_frame)
        if i > 100:
          break
    os.remove('test_out.mp4')

  duration = time.time() - start_time
  print(f' {i} frames in {duration:.2f}s ({i / duration:.2f} fps, {duration / i * 1000:.2f}ms/frame)')

def _video_transcode_rgb():
  video_reader = VideoReader(filename='test_files/dolphin_4096.mp4')

  codec_name, codec_options = utils.FindCodec(_CODEC_PREFERENCES)
  print(f' Using {codec_name}')

  @functools.partial(jax.jit, static_argnames=['frame_format'])
  def process_frame(raw_frame, frame_format: str) -> jnp.ndarray:
    frame_in = VideoReader.DecodeFrame(raw_frame, frame_format)
    return VideoWriter.EncodeFrame(frame_in)

  with VideoWriter(filename='test_out.mp4',
                   frame_rate=video_reader.frame_rate(),
                   pixfmt='yuv420p',
                   codec_name=codec_name,
                   codec_options=codec_options) as video_writer:
    for i, frame_data in enumerate(video_reader):
      raw_frame, frame_format = frame_data
      if i == 0:
        process_frame(raw_frame, frame_format).block_until_ready()
        start_time = time.time()
      video_writer.add_frame(encoded_frame=process_frame(raw_frame, frame_format))
      if i > 100:
        break
  os.remove('test_out.mp4')

  duration = time.time() - start_time
  print(f' {i} frames in {duration:.2f}s ({i / duration:.2f} fps, {duration / i * 1000:.2f}ms/frame)')

def _host_to_from_gpu():
  rng = np.random.default_rng()
  # This assumes yuv420.
  shapes = (('4k', 3840 * 2160 * 3 // 2), ('1080p', 1920 * 1080 * 3 // 2))
  for shape_name, shape in shapes:
    x = rng.integers(0, 1024, shape, dtype=np.uint16)
    cpu = jax.device_put(x, jax.devices('cpu')[0]).block_until_ready()
    start_time = time.time()
    ITERATIONS = 100
    for _ in range(ITERATIONS):
      jax.device_put(cpu, jax.devices('gpu')[0]).block_until_ready()
    duration = time.time() - start_time
    total_bytes = cpu.size * x.dtype.itemsize * ITERATIONS
    print(f' {shape_name} CPU => GPU {duration / ITERATIONS * 1000:.2f}ms/frame, {_pretty_size(total_bytes / duration)}/s')

    gpu = jax.device_put(x, jax.devices('gpu')[0]).block_until_ready()
    for _ in range(ITERATIONS):
      jax.device_put(gpu, jax.devices('cpu')[0]).block_until_ready()
    duration = time.time() - start_time
    total_bytes = cpu.size * x.dtype.itemsize * ITERATIONS
    print(f' {shape_name} GPU => CPU {duration / ITERATIONS * 1000:.2f}ms/frame, {_pretty_size(total_bytes / duration)}/s')


def _test_jax_op(fn, name='', iterations=100, input_shape=(3840, 2160, 3)):
  rng = np.random.default_rng()
  x = rng.random(input_shape, dtype=np.float32)
  x = jax.device_put(x).astype(FT()).block_until_ready()
  fn = jax.jit(fn)
  fn(x).block_until_ready()
  start_time = time.time()
  for i, _ in enumerate(range(iterations)):
    fn(x).block_until_ready()
    if time.time() - start_time > 1.0:
      break
  duration = time.time() - start_time
  print(f'{name} {duration / i * 1000:.2f}ms/frame')


def _pad_to_len(s: str, length: int = 80) -> str:
  padding = length - len(s) - 1
  return '>>>> ' + s + ' ' + '=' * padding

def _pretty_size(nbytes) -> str:
  nbytes = int(nbytes)
  if nbytes > (1024**4):
    return f'{nbytes / (1024**4):.2f} TB'
  if nbytes > (1024**3):
    return f'{nbytes / (1024**3):.2f} GB'
  if nbytes > (1024**2):
    return f'{nbytes / (1024**2):.2f} MB'
  if nbytes > (1024):
    return f'{nbytes / (1024):.2f} KB'
  return f'nbytes B'

def main():
  utils.EnablePersistentCache()

  default_device = jax.devices()[0]
  is_cpu = default_device.platform == 'cpu'
  print(f'Default device is: {default_device.platform} ({default_device.device_kind})')

  print(_pad_to_len('Video Decode (4k)'))
  _video_decode()
  print(_pad_to_len('Video Transcode (4k)'))
  _video_transcode()
  print(_pad_to_len('Video Transcode with RGB conversions (4k)'))
  _video_transcode_rgb()
  if not is_cpu:
    print(_pad_to_len('Host <=> GPU'))
    _host_to_from_gpu()
  print(_pad_to_len('Downsample 2x2 (linear)'))
  _test_jax_op(lambda x: jax_image.resize(x, (x.shape[0] // 2, x.shape[1] // 2, x.shape[2]), method='linear'), input_shape=(3840, 2160, 1))
  print(_pad_to_len('Downsample 2x2 (lanczos3)'))
  _test_jax_op(lambda x: jax_image.resize(x, (x.shape[0] // 2, x.shape[1] // 2, x.shape[2]), method='lanczos3'), input_shape=(3840, 2160, 1))
  print(_pad_to_len('Apply LUT'))
  _test_jax_op(lambda x: lut.apply_lut(x, _LUT_PATH))
  print(_pad_to_len('NL-means (fast)'))
  _test_jax_op(lambda x: nlmeans.nlmeans(img=x, strength=0.1, search_range=3, patch_size=7))

if __name__ == '__main__':
  main()
