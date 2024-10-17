import os
import platform
import sys

import jax
from jax import numpy as jnp
import numpy as np
import pytest

sys.path.append('.')

from JaxVidFlow import scale, utils

def test_downsample():
    img = jnp.array([
        0.0, 1.0, 0.5,
        1.0, 0.5, 0.0,
        1.0, 1.0, 1.0,
        0.0, 0.0, 0.0,
    ])
    img = jnp.reshape(img, (4, 3, 1))
    resized = scale.scale_image(img, new_width=6)
    assert resized.shape == (8, 6, 1)
    resized = scale.scale_image(img, new_height=12, multiple_of=4)
    assert resized.shape == (12, 8, 1)

    img = jnp.array([
        0.0, 0.5, 0.5, 1.0,
        0.25, 0.25, 1.0, 0.5,
        1.0, 1.0, 0.5, 0.5,
        0.5, 0.5, 0.0, 0.0,
    ])
    img = jnp.reshape(img, (4, 4, 1))
    resized = scale.scale_image(img, new_width=2)
    # This should be a simple averaging.
    expected = jnp.array([
        0.25, 0.75,
        0.75, 0.25,
    ])
    expected = jnp.reshape(expected, (2, 2, 1))
    np.testing.assert_allclose(resized, expected, atol=1e-4)

    # Non-integer.
    scaled = scale.scale_image(img, new_width=3, new_height=4, filter_method='linear')
    expected = jnp.array([
        0.15, 0.5,   0.85,
        0.25, 0.625, 0.65,
        1.0,  0.75,  0.5,
        0.5,  0.25,  0.0,
    ])
    expected = jnp.reshape(expected, (4, 3, 1))
    np.testing.assert_allclose(scaled, expected, atol=1e-4)

    # Upsample.
    img = jnp.reshape(jnp.array([
        0.25, 0.75,
        0.75, 0.25,
    ]), (2, 2, 1))
    expected = jnp.reshape(jnp.array([
        0.25, 0.5, 0.75,
        0.5,  0.5, 0.5,
        0.75, 0.5, 0.25,
    ]), (3, 3, 1))
    scaled = scale.scale_image(img, new_width=3, new_height=3, filter_method='linear')
    np.testing.assert_allclose(scaled, expected, atol=1e-4)