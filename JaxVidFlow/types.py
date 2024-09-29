import functools

import jax
from jax import numpy as jnp

@functools.cache
def FT():
  # Guess the best floating point type to use.
  default_device = jax.devices()[0]
  platform = default_device.platform

  if platform == 'cpu':
    # Use float32 on CPU for now. FP32 is only supported on x86 with AVX-512 FP16 extension, which is not widely
    # available as of this writing (2024). Not sure about the situation on ARM.
    return jnp.float32
  elif platform == 'gpu':
    # Use FP16 on GPUs. That's well supported, and on GPUs we are usually memory-BW-limited, so this makes things
    # much faster. The dynamic range and precision (10-bit fraction) should be enough for almost-lossless representation
    # in image applications.
    # This doubles the throughput for many GPU operations, and halves VRAM usage.
    return jnp.float16
  else:
    # What else can we be running on? TPUs? Use FP32 since it's most widely supported.
    return jnp.float32

