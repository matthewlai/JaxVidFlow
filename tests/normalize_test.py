import os
import platform
import sys

import jax
from jax import numpy as jnp
import numpy as np
import pytest

sys.path.append('.')

from JaxVidFlow import normalize

def test_calculate_gains():
    maxs = jnp.array([0.25, 0.5, 0.8])
    
    np.testing.assert_allclose(
        normalize._calculate_gains(
            channel_maxs=maxs, last_frame_gains=jnp.array([3.0, 4.0, 5.0]),
            last_frame_gains_valid=jnp.array(0.0), strength=1.0, temporal_smoothing=1.0,
            max_gain=10.0),
        np.array([4.0, 2.0, 1.25]), atol=1e-4)
    
    # With strength
    np.testing.assert_allclose(
        normalize._calculate_gains(
            channel_maxs=maxs, last_frame_gains=jnp.array([3.0, 4.0, 5.0]),
            last_frame_gains_valid=jnp.array(0.0), strength=0.5, temporal_smoothing=1.0,
            max_gain=10.0),
        np.array([2.5, 1.5, 1.125]), atol=1e-4)

    # With max gains.
    np.testing.assert_allclose(
        normalize._calculate_gains(
            channel_maxs=maxs, last_frame_gains=jnp.array([3.0, 4.0, 5.0]),
            last_frame_gains_valid=jnp.array(0.0), strength=1.0, temporal_smoothing=1.0,
            max_gain=3.0),
        np.array([3.0, 2.0, 1.25]), atol=1e-4)
    
    # With temporal smoothing (first frame).
    np.testing.assert_allclose(
        normalize._calculate_gains(
            channel_maxs=maxs, last_frame_gains=jnp.array([3.0, 4.0, 5.0]),
            last_frame_gains_valid=jnp.array(0.0), strength=1.0, temporal_smoothing=0.1,
            max_gain=3.0),
        np.array([3.0, 2.0, 1.25]), atol=1e-4)
    
    # With temporal smoothing (second frame).
    np.testing.assert_allclose(
        normalize._calculate_gains(
            channel_maxs=maxs, last_frame_gains=jnp.array([3.0, 4.0, 5.0]),
            last_frame_gains_valid=jnp.array(1.0), strength=0.5, temporal_smoothing=0.5,
            max_gain=10.0),
        np.array([2.75, 2.75, 3.0625]), atol=1e-4)

def test_normalize():
    img = jnp.reshape(jnp.array([
        0.25, 0.4, 0.1, 0.1,
        0.1, 0.25, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1
    ]), (4, 4, 1))
    
    # With no downsampling, we expect gain to be 1/0.4 = 2.5
    normalized_img, gains = normalize.normalize(
        img=img, last_frame_gains=jnp.array([0.0]),
        last_frame_gains_valid=jnp.array(0.0), strength=1.0, temporal_smoothing=1.0,
        max_gain=10.0, downsample_win=1)
    np.testing.assert_allclose(
        normalized_img,
        np.array(img) * 2.5, atol=1e-4)
    np.testing.assert_allclose(
        gains,
        np.array([2.5]), atol=1e-4)

    # With 2x2 downsampling, we expect gain to be 1/0.25 = 4.0
    normalized_img, gains = normalize.normalize(
        img=img, last_frame_gains=jnp.array([0.0]),
        last_frame_gains_valid=jnp.array(0.0), strength=1.0, temporal_smoothing=1.0,
        max_gain=10.0, downsample_win=2)
    
    np.testing.assert_allclose(
        normalized_img,
        np.clip(np.array(img) * 4.0, 0.0, 1.0), atol=1e-4)
    np.testing.assert_allclose(
        gains,
        np.array([4.0]), atol=1e-4)
