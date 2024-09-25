from typing import Sequence

from video_writer import VideoWriter

def FindCodec(candidates: Sequence[tuple[str, dict[str, str]]]) -> tuple[str, dict[str, str]]:
  codec_name = ''
  codec_options = None
  for codec_name, codec_options in candidates:
    if VideoWriter.test_codec(codec_name=codec_name):
      return codec_name, codec_options
  if not codec_name:
    raise RuntimeError(f'No valid codec found.') 