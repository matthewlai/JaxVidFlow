import functools
import sys
from typing import Sequence

import av
import jax
from jax import numpy as jnp
import numpy as np

sys.path.append('.')

from JaxVidFlow import colourspaces, compat, utils

class VideoWriter:
  # Note: Target bitrate is usually overridden by codec-specific constant quality control. One
  # exception is videotoolbox, which does support constant quality, but it uses the global quality scale
  # mechanism that's not yet implemented in PyAV.
  def __init__(self, filename: str, frame_rate: float, pixfmt: str,
         codec_name: str, codec_options: dict[str, str] | None, target_bitrate: int = 50000000):
    self.out_container = av.open(filename, 'w', options={'movflags': 'faststart'})
    self.out_video_stream = self.out_container.add_stream(
      codec_name=codec_name, rate=frame_rate, options=codec_options)
    self.out_audio_stream = None
    self.out_video_stream.pix_fmt = pixfmt
    self.out_codec_context = self.out_video_stream.codec_context

    # Sane default bit rate (usually overridden using codec-specific constant quality control).
    self.out_codec_context.bit_rate = target_bitrate

    # Hack for HEVC videos to play on Apple.
    if 'hevc' in codec_name:
      self.out_codec_context.codec_tag = 'hvc1'

    # When we write frames we delay by one to prevent a GPU sync.
    self.last_frame = None

    self.waiting_for_first_frame = True

    self._non_video_streams = []

  def add_frame(self, frame):
    """Add an RGB frame."""
    if self.waiting_for_first_frame:
      self.waiting_for_first_frame = False
      # Get width and height from the frame.
      width, height = frame.shape[1], frame.shape[0]

      self.out_video_stream.width = width
      self.out_video_stream.height = height

    if self.last_frame is not None:
      y, uv = self.last_frame
      y = np.array(y)
      uv = np.array(uv)

      frame_data_last = np.concatenate([
          y.reshape(-1),
          uv[:, :, 0].reshape(-1),
          uv[:, :, 1].reshape(-1)]).reshape(-1, y.shape[1])

      new_frame = av.VideoFrame.from_numpy_buffer(frame_data_last, format=self.frame_format())
      for packet in self.out_video_stream.encode(new_frame):
        self.out_container.mux(packet)

    self.last_frame = VideoWriter.EncodeFrame(frame) if frame is not None else None

  def frame_format(self) -> str:
    return self.out_video_stream.pix_fmt

  def write_audio_packets(self, audio_packets, in_audio_stream):
    if not self.out_audio_stream:
      self.out_audio_stream = self.out_container.add_stream(template=in_audio_stream)
    for packet in audio_packets:
      packet.stream = self.out_audio_stream
      self.out_container.mux(packet)

  def __enter__(self):
    return self

  def close(self):
    # Encode the last frame.
    self.add_frame(None)
    for packet in self.out_video_stream.encode():
      self.out_container.mux(packet)
    self.out_container.close()

  def __exit__(self, type, value, traceback):
    self.close()

  @staticmethod
  def test_codec(codec_name: str) -> bool:
    try:
      codec = av.codec.Codec(codec_name, mode='w')
      return True
    except av.codec.codec.UnknownCodecError:
      return False

  @staticmethod
  @functools.partial(jax.jit, static_argnames=['frame_format'])
  def EncodeFrame(rgb_frame: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], frame_format: str = 'yuv420p') -> jnp.ndarray:
    assert frame_format == 'yuv420p', f'Frame format {frame_format} not supported yet.'
    assert rgb_frame.shape[1] % 2 == 0, f'Frame width and height must be even for 4:2:0. ({rgb_frame.shape[1]}x{rgb_frame.shape[0]})'

    # First, RGB to YUV.
    yuv = colourspaces.RGB2YUV(rgb_frame)

    # Then we subsample U and V. Take upper left for now. This may or may not be standard, but close enough.
    # uv = yuv[0::2, 0::2, 1:]
    uv = compat.window_reduce_mean(yuv[:, :, 1:], (2, 2))

    # Convert to uint8 (TODO: add uint16 support for 10-bit)
    uv = jnp.round(uv * 255).astype(jnp.uint8)
    y = jnp.round(yuv[:, :, 0] * 255).astype(jnp.uint8)

    return y, uv
