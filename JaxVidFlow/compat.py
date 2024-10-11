import functools

import jax
from jax import numpy as jnp
import numpy as np

@functools.cache
def is_metal() -> bool:
  return jax.devices()[0].platform.lower() == 'metal'

def window_reduce_mean(x: jnp.ndarray, window: tuple[int]) -> jnp.ndarray:
  assert len(window) == 2
  assert x.shape[0] % window[0] == 0
  assert x.shape[1] % window[1] == 0
  if is_metal():
    # METAL backend doesn't support reduce_window https://github.com/jax-ml/jax/issues/21387
    # We can emulate it with a strided convolution.

    # Kernel in HWIO layout.
    i_ch = x.shape[2]
    kernel = jnp.zeros((window + (i_ch, i_ch)), dtype=x.dtype)
    ch_filter = jnp.ones(window, dtype=x.dtype) / np.prod(window)
    for dim in range(i_ch):
      kernel = kernel.at[:, :, dim, dim].set(ch_filter)

    # Image in NHWC layout.
    x = jnp.reshape(x, (1,) + x.shape)
    dn = jax.lax.conv_dimension_numbers(x.shape, kernel.shape, ('NHWC', 'HWIO', 'NHWC'))
    return jax.lax.conv_general_dilated(
      lhs=x, rhs=kernel, window_strides=window, padding='VALID', dimension_numbers=dn)[0, :, :, :]
  else:
    return jax.lax.reduce_window(x,
        init_value=0.0, computation=jax.lax.add,
        window_dimensions=window + (1,),
        window_strides=window + (1,), padding='valid') / np.prod(window)