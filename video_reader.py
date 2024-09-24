import functools

import av
import jax
from jax import image as jax_image
from jax import numpy as jnp

def undo_2x2subsample(x: jnp.ndarray) -> jnp.ndarray:
  # Undo subsampling (TODO: do this properly according to spec). Here we are assuming co-located with top left and
  # doing linear interpolation.
  
  # Approach 1:
  newshape = list(x.shape)
  newshape[-1] *= 2
  newshape[-2] *= 2
  return jax_image.resize(x, newshape, method='linear')

  # Approach 2 (just duplicate pixels - fast but not very good!):
  width, height = x.shape[-1] * 2, x.shape[-2] * 2
  x = jnp.repeat(x, repeats=2, axis=len(x.shape) - 1, total_repeat_length=width)
  x = jnp.repeat(x, repeats=2, axis=len(x.shape) - 2, total_repeat_length=height)
  return x


class VideoReader:
  def __init__(self, filename: str):
    self.in_container = av.open(filename)
    self.in_video_stream = self.in_container.streams.video[0]
    self.decode_generator = self.in_container.decode(video=0)

    # Enable frame threading.
    self.in_video_stream.thread_type = 'AUTO'

  def width(self) -> int:
    return self.in_video_stream.codec_context.width

  def height(self) -> int:
    return self.in_video_stream.codec_context.height

  def frame_rate(self) -> float:
    return self.in_video_stream.codec_context.rate

  def __iter__(self):
    return self

  def __next__(self) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], str]:
    """This returns a frame in native encoding and associated format. Use DecodeFrame() to decode into normalized RGB floats."""
    frame = next(self.decode_generator)
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

    return (y, u, v), frame.format.name

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
    y = y.astype(jnp.float32) / max_val
    u = u.astype(jnp.float32) / max_val - 0.5
    v = v.astype(jnp.float32) / max_val - 0.5

    u = undo_2x2subsample(u)
    v = undo_2x2subsample(v)

    assert y.shape == u.shape and u.shape == v.shape

    # Do BT.709 conversion to RGB.
    # https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.709_conversion

    r = y + 1.5748 * v
    g = y - 0.1873 * u - 0.4681 * v
    b = y + 1.8556 * u
    return jnp.clip(jnp.stack((r, g, b), axis=2), min=0.0, max=1.0)
