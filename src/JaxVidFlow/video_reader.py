import concurrent
from concurrent import futures
import dataclasses
import fractions
import functools
import logging
import math
import queue
import sys
import time
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
  max_val: int | float


@dataclasses.dataclass(order=True)
class PrioritizedEntry:
    priority: int
    item: Any = dataclasses.field(compare=False)

def _jax_array_from_video_plane(plane, dtype, values_per_pixel: int = 1) -> jnp.ndarray:
  total_line_size = abs(plane.line_size)
  bytes_per_pixel = jnp.dtype(dtype).itemsize * values_per_pixel
  useful_line_size = plane.width * bytes_per_pixel
  arr = jnp.frombuffer(plane, jnp.uint8)
  if total_line_size != useful_line_size:
    arr = arr.reshape(-1, total_line_size)[:, 0:useful_line_size].reshape(-1)
  return arr.view(jnp.dtype(dtype))

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

    self._reformatter = av.video.reformatter.VideoReformatter()

    # This holds futures for frame conversions. We have a priority queue because when seeking,
    # we need to be able to put one frame back into the front of the queue, and we do that by
    # having frame PTS as the priority.
    self._decoded_frames = queue.PriorityQueue()

    self._end_of_stream = threading.Event()

    self._audio_packets = queue.Queue()

    self._schedule_decode_task()

  def _check_and_decode_packet(self):
    if self._decoded_frames.qsize() < _MAX_FRAME_QUEUE_SIZE and not self._end_of_stream.is_set():
      try:
        packet = next(self.demux)
      except StopIteration:
        self._end_of_stream.set()
        return
      if packet.stream == self.in_audio_stream and packet.dts is not None:
        self._audio_packets.put(packet)
      if packet.stream == self.in_video_stream:
        start = time.time()
        for av_frame in packet.decode():
          # The decoder reuses frames, so we have to copy all the data out here.
          try:
            frame = VideoReader._convert_frame(
              av_frame=av_frame, width=self._width, height=self._height, jax_device=self._jax_device,
              reformatter=self._reformatter
            )
            self._decoded_frames.put(PrioritizedEntry(priority=av_frame.pts, item=frame))
          except Exception as e:
            print(e)

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
  def _convert_frame(av_frame, width, height, jax_device, reformatter) -> Frame:
    max_val = 0
    if av_frame.format.name in ('yuv420p', 'yuvj420p', 'nv12'):
      av_frame = reformatter.reformat(av_frame, width=width, height=height, format='yuv420p')
      dtype = jnp.uint8
      max_val = 2 ** 8 - 1
      assert len(av_frame.planes) == 3
      y, u, v = (_jax_array_from_video_plane(av_frame.planes[i], dtype) for i in range(3))
    elif av_frame.format.name in ('yuv420p10le'):
      av_frame = reformatter.reformat(av_frame, width=width, height=height)
      dtype = jnp.uint16
      max_val = 2 ** 10 - 1
      assert len(av_frame.planes) == 3
      y, u, v = (_jax_array_from_video_plane(av_frame.planes[i], dtype) for i in range(3))
    elif av_frame.format.name in ('yuv420p16le', 'p010le'):
      av_frame = reformatter.reformat(av_frame, width=width, height=height, format='yuv420p16le')
      dtype = jnp.uint16
      max_val = 2 ** 16 - 1
      assert len(av_frame.planes) == 3
      y, u, v = (_jax_array_from_video_plane(av_frame.planes[i], dtype) for i in range(3))
    else:
      raise RuntimeError(f'Unknwon frame format: {av_frame.format.name}')

    y, u, v = (jax.device_put(x, device=jax_device) for x in (y, u, v))

    # If we are scaling, we do it here using libav to minimise data transfer to the GPU. It's almost certainly not worth
    # the bandwidth to do the scaling on GPU. We do the conversion to RGB24 ourselves because we can do it faster than
    # ffmpeg even on CPU. Much faster on GPU. We also do it in floating point which is more accurate.
    # Unfortunately it looks like swscale is not thread-safe?
    # with reformatter_mutex:
    #   start = time.time()
    #   print(av_frame.format.name)
    #   # av_frame = reformatter.reformat(av_frame, width=width, height=height)
    #   # av_frame = reformatter.reformat(av_frame, width=width, height=height, format=format_to,
    #   #                                 dst_color_range=av.video.reformatter.ColorRange.JPEG)
    #   print(f'Reformat took {time.time() - start}')

    width = av_frame.width
    height = av_frame.height
    y = jnp.reshape(y, (height, width))
    u = jnp.reshape(u, (math.ceil(height / 2), math.ceil(width / 2)))
    v = jnp.reshape(v, (math.ceil(height / 2), math.ceil(width / 2)))

    rgb_data = VideoReader.ConvertToRGB((y, u, v), max_val)

    return Frame(
        data=rgb_data,
        frame_time=av_frame.time,
        rotation=av_frame.rotation,
        pts=av_frame.pts,
        max_val=1.0)

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
          self._decoded_frames.get(block=False)
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
      self._decoded_frames.put(PrioritizedEntry(priority=frame.pts, item=frame))

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
    frame = None

    while frame is None:
      try:
        prioritized_item = self._decoded_frames.get(timeout=0.1)
        frame = prioritized_item.item
      except queue.Empty:
        if self._end_of_stream.is_set():
          raise StopIteration()

    self._last_frame_time = frame.frame_time
    self._schedule_decode_task()

    return frame

  @staticmethod
  @functools.partial(jax.jit, static_argnames=['max_val'])
  def ConvertToRGB(raw_frame: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | jnp.ndarray, max_val: int) -> jnp.ndarray:
    y, u, v = raw_frame
    dtype = y.dtype
    y = y.astype(FT()) / max_val
    u = u.astype(FT()) / max_val
    v = v.astype(FT()) / max_val
    u = undo_2x2subsample(u)
    v = undo_2x2subsample(v)
    # U and V planes may be 1 pixel bigger than y in both directions, if y has odd width and/or height.
    if u.shape[0] == (y.shape[0] + 1):
      u = u[:-1, :]
      v = v[:-1, :]
    if u.shape[1] == (y.shape[1] + 1):
      u = u[:, :-1]
      v = v[:, :-1]
    assert y.shape == u.shape and u.shape == v.shape, f'{y.shape} {u.shape} {v.shape}'
    yuv = jnp.stack([y, u, v], axis=2)
    # Do BT.709 conversion to RGB.
    # https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.709_conversion
    rgb = colourspaces.YUV2RGB(yuv)
    return rgb
