import functools
import itertools
import os
import platform
import queue
import threading
import time
from typing import Generator

if platform.system() == 'Darwin':
	# Required for Jax on Metal (https://developer.apple.com/metal/jax/):
	os.environ['ENABLE_PJRT_COMPATIBILITY'] = '1'

import av
import jax
from jax import numpy as jnp
import numpy as np
from tqdm import tqdm

# Encoder test: ffmpeg -f lavfi -i testsrc=size=4096x3072:rate=30 -c:v h264_videotoolbox -f null -


def f32_to_uint8(x: jnp.ndarray) -> jnp.ndarray:
	assert x.dtype == jnp.float32
	return jnp.clip(jnp.round(x * 255), min=0.0, max=255.0).astype(jnp.uint8)


def uint8_to_f32(x: jnp.ndarray) -> jnp.ndarray:
	assert x.dtype == jnp.uint8
	return x.astype(jnp.float32) / 255


def yuv444p_to_rgbf(in_frame: jnp.ndarray) -> jnp.ndarray:
	height, width = in_frame.shape[1], in_frame.shape[2]
	in_frame = uint8_to_f32(in_frame)

	# u and v need to be shifted to be around 0
	y, u, v = in_frame[0], in_frame[1] - 0.5, in_frame[2] - 0.5
	r = y + 1.14075 * v
	g = y - 0.3455 * u - 0.7169 * v
	b = y + 1.7790 * u
	return jnp.clip(jnp.stack((r, g, b), axis=2), min=0.0, max=1.0)


@jax.jit
def process_frame(frame: jnp.ndarray) -> jnp.ndarray:
	frame = yuv444p_to_rgbf(frame)
	return f32_to_uint8(frame)


def main():
	in_container = av.open('test_files/dolphin_4096.mp4')

	in_video_stream = in_container.streams.video[0]

	# Enable frame threading.
	in_video_stream.thread_type = 'AUTO'

	out_container = av.open('test_out.mp4', 'w')

	out_codec = 'hevc_videotoolbox'

	in_width = in_video_stream.codec_context.width
	in_height = in_video_stream.codec_context.height

	# Copy audio stream over.
	out_video_stream = out_container.add_stream(codec_name=out_codec, rate=in_video_stream.codec_context.rate)
	out_video_stream.width = in_width
	out_video_stream.height = in_height
	out_video_stream.pix_fmt = 'yuv420p'

	start_time = time.time()

	frame_data_last = None

	for frame_i, frame in tqdm(enumerate(itertools.chain(in_container.decode(video=0), [None])), unit=' frames'):
		# Get a decoded frame.
		if frame is not None:
			frame_data_in = frame.to_ndarray(format='yuvj444p')

			# Submit a processing call to the GPU.
			frame_data_next = process_frame(frame_data_in)

		# Encode the last frame, if it exists. We delay the processing by one frame to not
		# force the GPU to be idle while we call encode.
		if frame_data_last is not None:
			new_frame = av.VideoFrame.from_ndarray(frame_data_last, format='rgb24')
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
