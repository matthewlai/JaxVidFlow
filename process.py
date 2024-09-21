import functools
import itertools
import os
import platform
import queue
import threading
import time
from typing import Any, Generator

if platform.system() == 'Darwin':
	# Required for Jax on Metal (https://developer.apple.com/metal/jax/):
	os.environ['ENABLE_PJRT_COMPATIBILITY'] = '1'

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import av
import jax
from jax import numpy as jnp
from jax import scipy as jsp
import numpy as np
from tqdm import tqdm


FTYPE = jnp.float32


# Encoders listed by preference.
_ENCODERS = [
	# Apple
	# Test: ffmpeg -i test_files/dolphin_4096.mp4 -c:v hevc_videotoolbox -f null -
	('hevc_videotoolbox', None),
	
	# NVIDIA
	# Test: ffmpeg -i test_files/dolphin_4096.mp4 -c:v hevc_nvenc -f null -
	('hevc_nvenc', None),

	('hevc', None),  # Software fallback
]


# Force x264 software H264 encoder at veryfast preset (for testing).
_FORCE_X264 = False
_X264_OPTIONS = {
	'preset': 'superfast',
	'crf': '24'
}


def f32_to_uint8(x: jnp.ndarray) -> jnp.ndarray:
	assert x.dtype == FTYPE
	return jnp.clip(jnp.round(x * 255), min=0.0, max=255.0).astype(jnp.uint8)

@functools.partial(jax.jit, static_argnames=['max_val'])
def uint_to_f32(x: jnp.ndarray, max_val: int) -> jnp.ndarray:
	assert x.dtype in (jnp.uint8, jnp.uint16)
	return x.astype(FTYPE) / max_val
	

def yuvf_to_rgbf(y: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
	assert y.dtype == FTYPE
	width, height = y.shape[1], y.shape[0]

	# Undo the subsampling if necessary.
	if u.shape[0] == (y.shape[0] // 2) and u.shape[1] == (y.shape[1] // 2):
		u = jnp.repeat(u, repeats=2, axis=1, total_repeat_length=width)
		u = jnp.repeat(u, repeats=2, axis=0, total_repeat_length=height)
		v = jnp.repeat(v, repeats=2, axis=1, total_repeat_length=width)
		v = jnp.repeat(v, repeats=2, axis=0, total_repeat_length=height)

	assert y.shape == u.shape and u.shape == v.shape
	
	# This assumes BT.709 colour space.
	# https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.709_conversion

	u -= 0.5
	v -= 0.5
	r = y + 1.5748 * v
	g = y - 0.1873 * u - 0.4681 * v
	b = y + 1.8556 * u
	return jnp.clip(jnp.stack((r, g, b), axis=2), min=0.0, max=1.0)

@jax.jit
def rgbf_to_yuvf(rgb: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
	assert rgb.dtype == FTYPE
	width, height = rgb.shape[1], rgb.shape[0]
	r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
	y = 0.2126 * r + 0.7152 * g + 0.0722 * b
	u = -0.1146 * r + -0.3854 * g + 0.5 * b
	v = 0.5 * r + -0.4542 * g + -0.0458 * b
	u += 0.5
	v += 0.5
	return tuple(jnp.clip(x, min=0.0, max=1.0) for x in (y, u, v))


def subsample_2x2(x: jnp.ndarray) -> jnp.ndarray:
	return x[0::2, 0::2]

@jax.jit
def rgbf_to_yuv420p_uint8(rgb: jnp.ndarray) -> jnp.ndarray:
	y, u, v = rgbf_to_yuvf(rgb)
	u = subsample_2x2(u)
	v = subsample_2x2(v)
	y, u, v = f32_to_uint8(y), f32_to_uint8(u), f32_to_uint8(v)
	return jnp.concatenate([y.reshape(-1), u.reshape(-1), v.reshape(-1)]).reshape(-1, y.shape[1])

@jax.jit
def normalize(rgb: jnp.ndarray) -> jnp.ndarray:
	max_r = jnp.max(rgb[:, :, 0]) + 0.001
	max_g = jnp.max(rgb[:, :, 1]) + 0.001
	max_b = jnp.max(rgb[:, :, 2]) + 0.001
	rgb = rgb.at[:, :, 0].multiply(1.0 / max_r)
	rgb = rgb.at[:, :, 1].multiply(1.0 / max_g)
	rgb = rgb.at[:, :, 2].multiply(1.0 / max_b)
	return rgb

@functools.partial(jax.jit, static_argnames=['bits'])
def process_frame(y: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray, bits: int) -> jnp.ndarray:
	y, u, v = (uint_to_f32(plane, max_val=2**bits - 1) for plane in (y, u, v))
	frame = yuvf_to_rgbf(y, u, v)
	frame = normalize(frame)
	return frame


def find_best_encoder() -> tuple[str, Any]:
	for option_name, option_config in _ENCODERS:
		try:
			codec = av.codec.Codec(option_name, mode='w')
			print(f'Found codec: {codec.long_name} ({codec.name})')
			return (option_name, option_config)
		except av.codec.codec.UnknownCodecError:
			continue


def main():
	in_container = av.open('test_files/dolphin_4096.mp4')

	in_video_stream = in_container.streams.video[0]

	# Enable frame threading.
	in_video_stream.thread_type = 'AUTO'

	out_container = av.open('test_out.mp4', 'w')

	if _FORCE_X264:
		out_codec, out_codec_options = 'h264', _X264_OPTIONS
	else:
		out_codec, out_codec_options = find_best_encoder()

	in_width = in_video_stream.codec_context.width
	in_height = in_video_stream.codec_context.height

	# Copy audio stream over.
	out_video_stream = out_container.add_stream(codec_name=out_codec, rate=in_video_stream.codec_context.rate, options=out_codec_options)
	out_video_stream.width = in_width
	out_video_stream.height = in_height
	out_video_stream.pix_fmt = 'yuv420p'
	out_codec_context = out_video_stream.codec_context

	start_time = time.time()

	frame_data_last = None

	for frame_i, frame in tqdm(enumerate(itertools.chain(in_container.decode(video=0), [None])), unit=' frames'):
		# Get a decoded frame.
		if frame is not None:
			bits = 0
			if frame.format.name in ('yuv420p', 'yuvj420p'):
				bits = 8
			elif frame.format.name in ('yuv420p10le'):
				bits = 10
			else:
				print(f'Unknwon frame format: {frame.format.name}')
				return
			dtype = jnp.uint8 if bits == 8 else jnp.uint16
			# Reading from video planes directly saves an extra copy in VideoFrame.to_ndarray.
			# Planes should be in machine byte order, which should also be what frombuffer() expects.
			y, u, v = (jnp.frombuffer(frame.planes[i], dtype) for i in range(3))
			width, height = frame.width, frame.height
			y = jnp.reshape(y, (height, width))
			u = jnp.reshape(u, (height // 2, width // 2))
			v = jnp.reshape(v, (height // 2, width // 2))

			# Submit a processing call to the GPU.
			frame_data_next = rgbf_to_yuv420p_uint8(process_frame(y, u, v, bits))

		# Encode the last frame, if it exists. We delay the processing by one frame to not
		# force the GPU to be idle while we call encode.
		if frame_data_last is not None:
			# Materialize the data into numpy at the last moment.
			frame_data_last = np.array(frame_data_last)
			new_frame = av.VideoFrame.from_numpy_buffer(frame_data_last, format='yuv420p')
			for packet in out_video_stream.encode(new_frame):
				out_container.mux(packet)

		# On the next iteration, we encode the frame that we have just submitted (once we already
		# have the next frame processing queued up).
		frame_data_last = frame_data_next

	# Flush the stream.
	for packet in out_video_stream.encode():
		out_container.mux(packet)

	out_container.close()

	duration = time.time() - start_time
	print(f'{frame_i} frames decoded, took {duration:.2f}s, FT: {duration/frame_i*1000:.2f} ms, {frame_i/duration:.3f} fps')


if __name__ == '__main__':
	main()
