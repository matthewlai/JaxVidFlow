import functools
import logging
import queue
import sys
from typing import Any, Sequence

import av
import jax
from jax import image as jax_image
from jax import numpy as jnp

sys.path.append('.')

from JaxVidFlow import colourspaces, scale
from JaxVidFlow.types import FT

logger = logging.getLogger(__name__)

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


class VideoReader:
  def __init__(self, filename: str, scale_width: int | None = None, scale_height: int | None = None):
    self.in_container = av.open(filename)
    self.in_video_stream = self.in_container.streams.video[0]
    self.in_audio_stream = self.in_container.streams.audio[0]
    self.demux = self.in_container.demux(video=0, audio=0)
    self._decoded_frames = queue.Queue()
    self._audio_packets = []

    side_data = self.in_video_stream.side_data
    self._rotation = 0
    if 'DISPLAYMATRIX' in side_data:
      self._rotation = side_data['DISPLAYMATRIX']

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

  def width(self) -> int:
    return self._width

  def height(self) -> int:
    return self._height

  def frame_rate(self) -> float:
    return self.in_video_stream.guessed_rate

  def audio_packets(self) -> Sequence[Any]:
    return self._audio_packets

  def clear_audio_packets(self):
    self._audio_packets = []

  def audio_stream(self):
    return self.in_audio_stream

  def __iter__(self):
    return self

  def __next__(self) -> tuple[jnp.ndarray, float]:
    """This returns a frame in normalized RGB and frame time."""

    # Decode is driven by the video stream. We always decode until we get some video frames, and just store the audio packets along
    # the way. At the end of the demuxing demux() will generate flushing packets (with packet.dts is None). We ignore those for audio.
    while self._decoded_frames.empty():
      packet = next(self.demux)
      if packet.stream == self.in_audio_stream and packet.dts is not None:
        self._audio_packets.append(packet)
      if packet.stream == self.in_video_stream:
        for frame in packet.decode():
          self._decoded_frames.put(frame)

    frame = self._decoded_frames.get()

    frame = frame.reformat(width=self._width, height=self._height)

    # Reading from video planes directly saves an extra copy in VideoFrame.to_ndarray.
    # Planes should be in machine byte order, which should also be what frombuffer() expects.
    bits = 0
    if frame.format.name in ('yuv420p', 'yuvj420p'):
      bits = 8
    elif frame.format.name in ('yuv420p10le'):
      bits = 10
    else:
      raise RuntimeError(f'Unknwon frame format: {frame.format.name}')
    dtype = jnp.uint8 if bits == 8 else jnp.uint16

    y, u, v = (jnp.frombuffer(frame.planes[i], dtype) for i in range(3))

    width = self.width()
    height = self.height()
    y = jnp.reshape(y, (height, width))
    u = jnp.reshape(u, (height // 2, width // 2))
    v = jnp.reshape(v, (height // 2, width // 2))

    return VideoReader.DecodeFrame((y, u, v), frame.format.name), frame.time, self._rotation

  @staticmethod
  @functools.partial(jax.jit, static_argnames=['frame_format'])
  def DecodeFrame(raw_frame: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | jnp.ndarray, frame_format: str) -> jnp.ndarray:
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
