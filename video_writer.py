import functools

import av
import jax
from jax import numpy as jnp
import numpy as np

class VideoWriter:
  def __init__(self, filename: str, width: int, height: int, frame_rate: float, pixfmt: str,
         codec_name: str, codec_options: dict[str, str] | None):
    self.out_container = av.open(filename, 'w')
    self.out_video_stream = self.out_container.add_stream(
      codec_name=codec_name, rate=frame_rate, options=codec_options)
    self.out_video_stream.width = width
    self.out_video_stream.height = height
    self.out_video_stream.pix_fmt = pixfmt
    self.out_codec_context = self.out_video_stream.codec_context

    # When we write frames we delay by one to prevent a GPU sync.
    self.last_frame = None

  def add_frame(self, encoded_frame: jnp.ndarray | np.ndarray | None):
    """Add a raw frame already encoded using EncodeFrame()."""
    if self.last_frame is not None:
      frame_data_last = np.array(self.last_frame)
      new_frame = av.VideoFrame.from_numpy_buffer(frame_data_last, format=self.frame_format())
      for packet in self.out_video_stream.encode(new_frame):
        self.out_container.mux(packet)

    self.last_frame = encoded_frame

  def frame_format(self) -> str:
    return self.out_video_stream.pix_fmt

  def __enter__(self):
    return self

  def __exit__(self, type, value, traceback):
    # Encode the last frame.
    self.add_frame(None)
    for packet in self.out_video_stream.encode():
      self.out_container.mux(packet)
    self.out_container.close()

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
    r, g, b = rgb_frame[:, :, 0], rgb_frame[:, :, 1], rgb_frame[:, :, 2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    u = -0.1146 * r + -0.3854 * g + 0.5 * b
    v = 0.5 * r + -0.4542 * g + -0.0458 * b
    u += 0.5
    v += 0.5
    y = jnp.clip(y, min=0.0, max=1.0)

    # Then we subsample U and V. Take upper left for now. This may or may not be standard, but close enough.
    u, v = u[0::2, 0::2], v[0::2, 0::2]
    u = jnp.clip(u, min=0.0, max=1.0)
    v = jnp.clip(v, min=0.0, max=1.0)

    # Convert to uint8 (TODO: add uint16 support for 10-bit).
    y = jnp.round(y * 255).astype(jnp.uint8)
    u = jnp.round(u * 255).astype(jnp.uint8)
    v = jnp.round(v * 255).astype(jnp.uint8)

    # Finally, concatenate into the packed format libAV wants.
    return jnp.concatenate([y.reshape(-1), u.reshape(-1), v.reshape(-1)]).reshape(-1, y.shape[1])
