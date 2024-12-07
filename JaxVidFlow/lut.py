import functools
import io
import logging
import jax
from jax import numpy as jnp
import numpy as np

from JaxVidFlow import compat

logger = logging.getLogger(__name__)

def _lut_impl(lower: jnp.ndarray, higher: jnp.ndarray, frac: jnp.ndarray, lut: jnp.ndarray) -> jnp.ndarray:
  # Finally, the interpolated value is a linear combination of the colours at
  # the lower index and the higher index, weighted by frac.
  # The CUBE format is indexed in [b, g, r].
  lut_at_lower = lut[lower[:, :, 2], lower[:, :, 1], lower[:, :, 0]]
  lut_at_higher = lut[higher[:, :, 2], higher[:, :, 1], higher[:, :, 0]]
  return lut_at_higher * frac + lut_at_lower * (1.0 - frac)

@functools.partial(jax.jit, static_argnames=['lut_path'])
def apply_lut(frame: jnp.ndarray, lut_path: str) -> jnp.ndarray:
  assert not compat.is_metal(), "This implementation crashes METAL. See https://github.com/jax-ml/jax/issues/23902"

  title = ''
  size = 0
  entries = []
  with open(lut_path, 'r') as lut_file:
    for line in lut_file.readlines():
      if line.startswith('TITLE'):
        title = line[7:-2]
      elif line.startswith('LUT_3D_SIZE'):
        size = int(line[12:])
      else:
        parts = line.split()
        if len(parts) == 3:
          entries.append(np.array([float(x) for x in parts]))
  if size == 0:
    raise RuntimeError('No LUT_3D_SIZE in LUT file.')
  logger.info(f'Using {lut_path} ({title}) Size: {size}')
  assert len(entries) == (size * size * size)

  lut = jnp.array(np.stack(entries).reshape(size, size, size, 3))

  # First we compute the floating point indices of each pixel in the LUT
  scaled_frame = frame * (lut.shape[0] - 1)

  # Then we get the floor and ceil indices for all 3 colours.
  lower = jnp.floor(scaled_frame).astype(jnp.uint16)
  higher = jnp.ceil(scaled_frame).astype(jnp.uint16)

  # Now we determine the fractional part along each colour.
  frac = scaled_frame - lower

  lut_at_lower = lut[lower[:, :, 2], lower[:, :, 1], lower[:, :, 0]]
  lut_at_higher = lut[higher[:, :, 2], higher[:, :, 1], higher[:, :, 0]]
  return lut_at_higher * frac + lut_at_lower * (1.0 - frac)
