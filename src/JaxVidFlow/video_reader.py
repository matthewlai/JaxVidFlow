import concurrent
from concurrent import futures
import dataclasses
import fractions
import functools
import logging
import math
import queue
import sys
import threading
from typing import Any, Sequence

import av
import jax
from jax import image as jax_image
from jax import numpy as jnp

sys.path.append('.')

from JaxVidFlow import colourspaces, scale
from JaxVidFlow.types import FT

logger = logging.getLogger(__name__)

_MIN_SEEK_TIME = 2.0  # If we are seeking ahead by less than this amount, just keep decoding. 2.0 is a common keyframe interval.

# How many frames to decode/convert ahead. Note that this is a suggestion. If we get a packet with a lot of frames, we have to
# decode them all to avoid a deadlock trying to stop decoding threads.
_MAX_FRAME_QUEUE_SIZE = 4

def undo_2x2subsample(x: jnp.ndarray) -> jnp.ndarray:
  # Undo subsampling (TODO: do this properly according to spec). Here we are assuming co-located with top left and
  # doing linear interpolation.
  
  # Approach 1:
  # newshape = list(x.shape)
  # newshape[-1] *= 2
  # newshape[-2] *= 2
  # return jax_image.resize(x, newshape, method='linear')

  # Approach 2 (just duplicate pixels - fast but not very good!):
  width, height = x.shape[-1] * 2, x.shape[-2] * 2
  x = jnp.repeat(x, repeats=2, axis=len(x.shape) - 1, total_repeat_length=width)
  x = jnp.repeat(x, repeats=2, axis=len(x.shape) - 2, total_repeat_length=height)
  return x


@dataclasses.dataclass
class Frame:
  data: jnp.ndarray
  frame_time: float
  rotation: int
  pts: int


@dataclasses.dataclass(order=True)
class PrioritizedEntry:
    priority: int
    item: Any = dataclasses.field(compare=False)


class VideoReader:
  def __init__(self, filename: str, scale_width: int | None = None, scale_height: int | None = None,
               hwaccel: av.codec.hwaccel.HWAccel | str | None = None, jax_device: Any = None, max_workers: int | None = None):
    self._hwaccel = None
    if isinstance(hwaccel, str):
      self._hwaccel = av.codec.hwaccel.HWAccel(device_type=hwaccel)
    elif isinstance(hwaccel, av.codec.hwaccel.HWAccel):
      self._hwaccel = hwaccel

    if jax_device is None:
      jax_device = jax.devices()[0]
    self._jax_device = jax_device

    self.in_container = av.open(filename, hwaccel=self._hwaccel)
    self.in_video_stream = self.in_container.streams.video[0]
    self.in_audio_stream = self.in_container.streams.audio[0]
    self.demux = self.in_container.demux(video=0, audio=0)

    logger.debug('Streams:')
    for i, stream in enumerate(self.in_container.streams):
      logger.debug(f'  {i}: {stream.type}')
      if isinstance(stream, av.video.stream.VideoStream):
        codec_context = stream.codec_context
        logger.debug(f'    {stream.format.width}x{stream.format.height}@{stream.guessed_rate}fps'
                     f' ({stream.codec.name},{codec_context.pix_fmt})')
      if isinstance(stream, av.audio.stream.AudioStream):
        codec_context = stream.codec_context
        logger.debug(f'    {stream.codec.name} {codec_context.sample_rate}Hz {codec_context.layout.name}')
      if isinstance(stream, av.data.stream.DataStream):
        codec_context = stream.codec_context
        logger.debug(f'    {stream.name}')
        # We don't know how to copy data streams.

    # Enable frame threading.
    self.in_video_stream.thread_type = 'AUTO'

    self._width = self.in_video_stream.codec_context.width
    self._height = self.in_video_stream.codec_context.height

    self._height, self._width = scale.calculate_new_dims(
        old_width=self._width, old_height=self._height,
        multiple_of=8, new_width=scale_width, new_height=scale_height)

    # Frame time of the last frame (that has been extracted from the decoded frames queue).
    self._last_frame_time = 0.0

    # Our decoding process has 3 stages -
    # 1. Getting a packet from the demuxer
    # 2. Decode 0 or more frames from the packet
    # 3. Convert each frame into RGBF32 on Jax device.
    #
    # 1 and 2 must happen serially from our perspective (libav internally does multi-threaded
    # decode). So we have an executor with max_workers=1 for this. Step 3 has another executor
    # that converts the frames in parallel.

    self._decode_executor = futures.ThreadPoolExecutor(max_workers=1)

    self._convert_executor = futures.ThreadPoolExecutor(max_workers=max_workers)

    # This holds futures for frame conversions. We have a priority queue because when seeking,
    # we need to be able to put one frame back into the front of the queue, and we do that by
    # having frame PTS as the priority.
    self._converted_frames = queue.PriorityQueue()

    self._end_of_stream = threading.Event()

    self._audio_packets = queue.Queue()

    self._schedule_decode_task()

  def _check_and_decode_packet(self):
    if self._converted_frames.qsize() < _MAX_FRAME_QUEUE_SIZE and not self._end_of_stream.is_set():
      try:
        packet = next(self.demux)
      except StopIteration:
        self._end_of_stream.set()
        return
      if packet.stream == self.in_audio_stream and packet.dts is not None:
        self._audio_packets.put(packet)
      if packet.stream == self.in_video_stream:
        for av_frame in packet.decode():
          frame_future = self._convert_executor.submit(VideoReader._convert_frame, av_frame, self._width, self._height, self._jax_device)
          self._converted_frames.put(PrioritizedEntry(priority=av_frame.pts, item=frame_future))

      # Schedule the next task. This will silently fail if the threadpool is getting shutdown (eg for seeking).
      # All other checks will happen when the task gets run, so we don't need to repeat them here.
      self._schedule_decode_task()

  def _schedule_decode_task(self):
    try:
      self._decode_executor.submit(VideoReader._check_and_decode_packet, self)
    except RuntimeError:
      # We are shutting down.
      pass

  @staticmethod
  def _convert_frame(av_frame, width, height, jax_device) -> Frame:
    # Note that this runs in a worker thread, so should not access self.
    # Reading from video planes directly saves an extra copy in VideoFrame.to_ndarray.
    # Planes should be in machine byte order, which should also be what frombuffer() expects.
    bits = 0
    if av_frame.format.name in ('yuv420p', 'yuvj420p', 'nv12'):
      bits = 8
      format_to = 'yuv420p'
    elif av_frame.format.name in ('yuv420p10le', 'p010le'):
      bits = 10
      format_to = 'yuv420p10le'
    else:
      raise RuntimeError(f'Unknwon frame format: {av_frame.format.name}')
    dtype = jnp.uint8 if bits == 8 else jnp.uint16

    # If we are scaling, we do it here using libav to minimise data transfer to the GPU. It's almost certainly not worth
    # the bandwidth to do the scaling on GPU. We do the conversion to RGB24 ourselves because we can do it faster than
    # ffmpeg even on CPU. Much faster on GPU. We also do it in floating point which is more accurate.
    av_frame = av_frame.reformat(width=width, height=height, format=format_to)

    y, u, v = (jax.device_put(jnp.frombuffer(av_frame.planes[i], dtype), device=jax_device) for i in range(3))

    y = jnp.reshape(y, (height, width))
    u = jnp.reshape(u, (height // 2, width // 2))
    v = jnp.reshape(v, (height // 2, width // 2))

    return Frame(
        data=VideoReader.ConvertToRGB((y, u, v), av_frame.format.name),
        frame_time=av_frame.time,
        rotation=av_frame.rotation,
        pts=av_frame.pts)

  def width(self) -> int:
    return self._width

  def height(self) -> int:
    return self._height

  def set_width(self, width) -> None:
    self._width = width

  def set_height(self, height) -> None:
    self._height = height

  def frame_rate(self) -> fractions.Fraction:
    return self.in_video_stream.guessed_rate

  def num_frames(self) -> int:
    return self.in_video_stream.frames

  def duration(self) -> float:
    return float(self.in_video_stream.duration * self.in_video_stream.time_base)

  def seek(self, desired_frame_time):
    # Move the decoder so that __next__() returns the frame closest to the desired_frame_time.
    # Note that seeking currently get audio out of sync.
    offset = math.floor(desired_frame_time / self.in_video_stream.time_base)
    should_seek = False
    if desired_frame_time < self._last_frame_time:
      # Always seek backwards.
      should_seek = True
    elif (desired_frame_time - self._last_frame_time) > _MIN_SEEK_TIME:
      should_seek = True

    if should_seek:
      # Now we need to clear the queue and start decoding again after seek.
      self._decode_executor.shutdown(cancel_futures=True)
      while True:
        try:
          self._converted_frames.get(block=False)
        except queue.Empty:
          break

      while True:
        try:
          self._audio_packets.get(block=False)
        except queue.Empty:
          break

      self.in_container.seek(offset=offset, stream=self.in_video_stream)

      # After seeking we need to get a new demux because the last one may have already hit EOF and exited.
      self.demux = self.in_container.demux(video=0, audio=0)
      self._end_of_stream.clear()
      self._decode_executor = futures.ThreadPoolExecutor(max_workers=1)
      self._schedule_decode_task()

    # Here we just keep popping frames until we get to the right time. This is
    # inefficient because we actually don't need to do the conversion for just seeking,
    # but this keeps the code much simpler.
    current_frame_time = None
    frame = None
    while current_frame_time is None or current_frame_time < desired_frame_time:
      try:
        frame = self.__next__()
        current_frame_time = frame.frame_time
      except StopIteration:
        break

    if frame is None:
      self._end_of_stream.set()
    else:
      self._end_of_stream.clear()
      # Here we put the last frame back.
      frame_future = futures.Future()
      frame_future.set_result(frame)
      self._converted_frames.put(PrioritizedEntry(priority=frame.pts, item=frame_future))

  def audio_packets(self) -> Sequence[Any]:
    ret = []
    while not self._audio_packets.empty():
      ret.append(self._audio_packets.get())
    return ret

  def audio_stream(self):
    return self.in_audio_stream

  def __iter__(self):
    return self

  def __next__(self) -> tuple[jnp.ndarray, float]:
    """This returns a frame in normalized RGB and frame time."""
    frame_future = None

    while frame_future is None:
      try:
        prioritized_item = self._converted_frames.get(timeout=0.1)
        frame_future = prioritized_item.item
      except queue.Empty:
        if self._end_of_stream.is_set():
          raise StopIteration()

    frame = frame_future.result()
    self._last_frame_time = frame.frame_time
    self._schedule_decode_task()
    return frame

  @staticmethod
  @functools.partial(jax.jit, static_argnames=['frame_format'])
  def ConvertToRGB(raw_frame: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | jnp.ndarray, frame_format: str) -> jnp.ndarray:
    bits = 0
    if frame_format in ('yuv420p', 'yuvj420p'):
      bits = 8
    elif frame_format in ('yuv420p10le'):
      bits = 10
    else:
      raise RuntimeError(f'Unknwon frame format: {frame_format}')

    y, u, v = raw_frame

    max_val = 2 ** bits - 1
    y = y.astype(FT()) / max_val
    u = u.astype(FT()) / max_val
    v = v.astype(FT()) / max_val

    u = undo_2x2subsample(u)
    v = undo_2x2subsample(v)

    assert y.shape == u.shape and u.shape == v.shape

    yuv = jnp.stack([y, u, v], axis=2)

    # Do BT.709 conversion to RGB.
    # https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.709_conversion

    rgb = colourspaces.YUV2RGB(yuv)

    return jnp.clip(rgb, min=0.0, max=1.0)
