import dataclasses

# Fast H264 CPU encoder for testing.
X264_ENCODER = 'x264'
X264_OPTIONS = {
  'preset': 'superfast',
  'crf': '24'
}

@dataclasses.dataclass
class Config:
  # Encoders listed by preference with options.
  encoders: list[tuple[str, dict[str, str] | None]] = dataclasses.field(
    default_factory=lambda: [
      # Test: ffmpeg -i test_files/dolphin_4096.mp4 -c:v {encoder_name} -f null -

      (
        'hevc_videotoolbox',
        # Videotoolbox supports constant quality mode on Apple Silicon, but uses the global quality scale
        # mechanism that doesn't seem to be supported in PyAV yet, so we have to just use bit rate control for now.
        {
        }
      ),
      
      # NVIDIA
      ( 
        'hevc_nvenc',
        {
          # CQ=23 seems like a reasonable default. ~60mbps for city streets at 4K.
          'rc': 'vbr_hq',
          'cq': '23',
          'qmin': '23',
          'qmax': '23',
        }
      ),

      # TODO: Add AMD and Intel encoders.

      # Software fallback.
      (
        'hevc',
        {
          'crf': '20'
        }
      ),
  ])

  force_cpu_backend: bool = False
  profiling: bool = False

