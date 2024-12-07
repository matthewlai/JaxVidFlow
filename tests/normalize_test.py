import os
import platform
import sys

import jax
from jax import numpy as jnp
import numpy as np
import pytest

sys.path.append('.')

from JaxVidFlow import normalize

# def normalize(img: jnp.ndarray, last_frame_mins: jnp.ndarray | None, last_frame_maxs: jnp.ndarray | None,
#               temporal_smoothing: float = 0.05, max_gain: float = 10.0,
#               downsample_win: int = 8) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

def test_normalize():
    img = jnp.reshape(jnp.array([
        0.25, 0.4, 0.1, 0.1,
        0.1, 0.25, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1
    ]), (4, 4, 1))
    
    normalized_img, mins, maxs = normalize.normalize(
        img=img, last_frame_mins=None, last_frame_maxs=None,
        temporal_smoothing=1.0, max_gain=10.0, downsample_win=1)
    np.testing.assert_allclose(
        maxs,
        np.array(0.4), atol=1e-4)
    np.testing.assert_allclose(
        mins,
        np.array(0.1), atol=1e-4)

    img_exp = jnp.reshape(jnp.array([
        0.5, 1.0, 0.0, 0.0,
        0.0, 0.5, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    ]), (4, 4, 1))
    np.testing.assert_allclose(
        normalized_img,
        img_exp, atol=1e-4)

    # Force a max gain of 3.0 (instead of 3.33)
    normalized_img, mins, maxs = normalize.normalize(
        img=img, last_frame_mins=None, last_frame_maxs=None,
        temporal_smoothing=1.0, max_gain=3.0, downsample_win=1)
    np.testing.assert_allclose(
        maxs,
        np.array(0.4), atol=1e-4)
    np.testing.assert_allclose(
        mins,
        np.array(0.1), atol=1e-4)

    img_exp = jnp.reshape(jnp.array([
        0.45, 0.9, 0.0, 0.0,
        0.0, 0.45, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    ]), (4, 4, 1))
    np.testing.assert_allclose(
        normalized_img,
        img_exp, atol=1e-4)

    # With some temporal smoothing.
    normalized_img, mins, maxs = normalize.normalize(
        img=img, last_frame_mins=jnp.array([0.05]), last_frame_maxs=jnp.array([0.5]),
        temporal_smoothing=0.1, max_gain=10.0, downsample_win=1)
    np.testing.assert_allclose(
        maxs,
        np.array(0.49), atol=1e-4)
    np.testing.assert_allclose(
        mins,
        np.array(0.055), atol=1e-4)

    img_exp = jnp.reshape(jnp.array([
        0.448276, 0.7931, 0.103448, 0.103448,
        0.103448, 0.448276, 0.103448, 0.103448,
        0.103448, 0.103448, 0.103448, 0.103448,
        0.103448, 0.103448, 0.103448, 0.103448
    ]), (4, 4, 1))
    np.testing.assert_allclose(
        normalized_img,
        img_exp, atol=1e-4)