import argparse
import functools
import logging
import math
import os
import pathlib
import platform
import queue
import sys
import threading
import time
from typing import Any, Generator, Sequence

import psutil

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
from PIL import Image
from tqdm import tqdm

sys.path.append('.')

from JaxVidFlow import colourspaces, compat, gyroflow, lut, nlmeans, normalize, scale, utils
from JaxVidFlow.config import Config
from JaxVidFlow.types import FT
from JaxVidFlow.video_reader import VideoReader
from JaxVidFlow.video_writer import VideoWriter


arg_parser = argparse.ArgumentParser(
    description='Stabilizes and colour corrects dive videos')

arg_parser.add_argument('path', help='Either a single video or a directory', type=str)
arg_parser.add_argument('--output_dir', help='Output dir relative to file input dir', default='processed', type=str)
arg_parser.add_argument('--comparison', help='Create comparison videos', default=False, type=bool)

GYROFLOW_PRESET = '{ "light_refraction_coefficient": 1.33 }'


logger = logging.getLogger(__name__)
logging.basicConfig(encoding='utf-8', level=logging.INFO,
                    format='%(asctime)s.%(msecs)04d:%(filename)s:%(funcName)s:%(lineno)s:%(levelname)s: %(message)s',)

# Approximately equivalent to for benchmarking purposes:
# ffmpeg -i test_files/lionfish.mp4 -vf "scale=1920:-1,normalize,format=yuv420p" -c:v hevc_videotoolbox -f null -
@functools.partial(jax.jit, static_argnames=['output_for_gyroflow', 'rotation'])
def process_frame(frame, carry, output_for_gyroflow: bool, rotation: int) -> tuple[jnp.ndarray, jnp.ndarray]:
  if carry is None:
    last_frame_mins = None
    last_frame_maxs = None
  else:
    last_frame_mins, last_frame_maxs = carry

  assert rotation == 0 or not output_for_gyroflow, 'Gyroflow cannot handle rotated videos yet'

  if rotation != 0:
    assert rotation % 90 == 0
    times = rotation // 90
    frame = jnp.rot90(frame, k=times)

  ref = frame

  # Theoretically we should be able to go to Rec 709 (in float), then to linear, and normalize there. But at least
  # with the DJI log, it really crashes red, and doesn't look good in practice.
  # We do the normalization in log space instead, and optionally apply gamma correction into Rec709.
  # This doesn't make much mathematical sense, but produces aesthetically pleasing results.
  # frame = lut.apply_lut(frame, 'luts/D_LOG_M_to_Rec_709_LUT_ZG_Rev1.cube')
  frame, last_frame_mins, last_frame_maxs = normalize.normalize(
      img=frame, last_frame_mins=last_frame_mins, last_frame_maxs=last_frame_maxs, max_gain=10.0, downsample_win=16,
      temporal_smoothing=0.0)
  frame = jnp.pow(frame, 1.1)
  frame_out = frame
  if output_for_gyroflow:
    # Gyroflow will do the rotation.
    frame_out = gyroflow.to_gyroflow(frame_out)
  return frame_out, ref, (last_frame_mins, last_frame_maxs)


def calculate_nlmeans_params(raw_frame, frame_format: str) -> frozenset[str, int]:
  frame_in = VideoReader.DecodeFrame(raw_frame, frame_format)
  frame = colourspaces.Rec709ToLinear(frame_in)
  print(f'First frame sigma: {np.array(utils.EstimateNoiseSigma(frame))}')
  return frozenset(nlmeans.default_nlmeans_params(frame).items())

def process_video(input_path: str, output_path: str, codec_name, codec_options, comparison_out):
  video_reader = VideoReader(filename=input_path, scale_width=1920)

  analysis_file = gyroflow.gyroflow_create_project_file(video_path=input_path, preset=GYROFLOW_PRESET)

  gyroflow_filter = gyroflow.Gyroflow(
      gyroflow_project_path=analysis_file,
      gyroflow_lib_path='gyroflow/libgyroflow_frei0r.dylib')

  sharding = None

  comparison_video_writer = None

  if comparison_out is not None:
    comparison_video_writer = VideoWriter(
        filename=comparison_out,
        frame_rate=video_reader.frame_rate(),
        pixfmt='yuv420p',
        codec_name=codec_name,
        codec_options=codec_options)

  with VideoWriter(filename=output_path,
                   frame_rate=video_reader.frame_rate(),
                   pixfmt='yuv420p',
                   codec_name=codec_name,
                   codec_options=codec_options) as video_writer:
    carry = None
    for frame_i, frame_data in tqdm(enumerate(video_reader), unit=' frames'):
      frame, frame_time, rotation = frame_data

      if sharding is None:
        sharding = utils.GetSharding(num_devices_limit=None, divisor_of=frame.shape[0])
      frame_in = jax.device_put(frame, sharding)

      # Gyroflow can't handle rotated videos yet.
      use_gyroflow = rotation == 0

      # Submit a processing call to the GPU.
      frame, ref, carry = process_frame(frame=frame_in, carry=carry, output_for_gyroflow=use_gyroflow, rotation=rotation)

      if use_gyroflow:
        frame = gyroflow_filter.process_frame(frame=frame, frame_time=frame_time, rotation=rotation)

        # Gyroflow delays by one frame, and returns None for the first frame. This implementation
        # drops the last frame.
        if frame is None:
          continue

        frame = gyroflow.from_gyroflow(frame)

      video_writer.add_frame(frame=frame)
      video_writer.write_audio_packets(audio_packets=video_reader.audio_packets(),
                                       in_audio_stream=video_reader.audio_stream())
      if comparison_video_writer:
        comparison_frame = utils.MergeSideBySide(frame, ref)
        comparison_video_writer.add_frame(frame=comparison_frame)
        comparison_video_writer.write_audio_packets(audio_packets=video_reader.audio_packets(),
                                                    in_audio_stream=video_reader.audio_stream())

      video_reader.clear_audio_packets()

  if comparison_video_writer:
    comparison_video_writer.close()

  os.unlink(analysis_file)

def main(args):
  # Enable the persistent compilation cache so we are not recompiling every execution.
  utils.EnablePersistentCache()

  # Get default configs.
  config = Config(force_cpu_backend=False, profiling=False)

  if config.force_cpu_backend:
    jax.config.update('jax_platform_name', 'cpu')

  codec_name, codec_options = utils.FindCodec(config.encoders)

  print(f'Using {codec_name} ({codec_options})')

  user_input_path = pathlib.Path(args.path).resolve().absolute()

  input_paths = []

  if user_input_path.is_file():
    input_paths.append(user_input_path)
  elif user_input_path.is_dir():
    input_paths = list(user_input_path.glob('*.[Mm][Pp]4'))
  else:
    raise ValueError(f'Not found: {user_input_path}')

  mappings = []

  for input_path in input_paths:
    parent = input_path.parent
    output_dir = pathlib.Path(parent, args.output_dir)
    output_filename = pathlib.Path(str(input_path.stem) + '.mp4')
    output_path = pathlib.Path(output_dir, output_filename)
    comparison_path = None
    if args.comparison:
      comparison_path = pathlib.Path(output_dir, 'compare', output_filename)
    mappings.append((str(input_path), str(output_path), comparison_path))

  print(mappings)

  if not mappings:
    raise ValueError('No video found')

  print(f'{len(mappings)} videos found -')
  parent_path = str(pathlib.Path(mappings[0][0]).parent) + '/'
  print(f'{parent_path}...')
  for input_path, output_path, _ in mappings:
    print(f'\t{pathlib.Path(input_path).name} => {str(pathlib.Path(output_path)).replace(parent_path, "")}')

  if config.profiling:
    jax.profiler.start_trace("/tmp/jax-trace", create_perfetto_link=True)

  pathlib.Path(parent, args.output_dir).mkdir(exist_ok=True)

  if args.comparison:
    pathlib.Path(parent, args.output_dir, pathlib.Path('compare')).mkdir(exist_ok=True)

  for vid_i, (input_path, output_path, comparison_out) in enumerate(mappings):
    print(f'Processing {input_path} ({vid_i + 1} / {len(mappings)})')
    process_video(input_path, output_path, codec_name, codec_options, comparison_out=comparison_out)

  if config.profiling:
    jax.profiler.stop_trace()

if __name__ == '__main__':
  main(arg_parser.parse_args())
