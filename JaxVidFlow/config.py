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

      # Apple (TODO: Find good default params)
      ('hevc_videotoolbox', None),
      
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

      # Software fallback (library defaults are as good as any)
      ('hevc', None),
  ])

  force_cpu_backend: bool = False
  profiling: bool = False

