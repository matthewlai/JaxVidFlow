import os
import platform
import sys

import jax
from jax import numpy as jnp
import numpy as np
import pytest

sys.path.append('.')

from JaxVidFlow import nlmeans, utils

utils.EnablePersistentCache()

def _add_noise(img: np.ndarray, sigma: np.ndarray):
	noise = np.random.normal(size=img.shape)
	noise *= sigma  # This can be per-channel or just one sigma for all channels.
	return np.clip(img + noise, 0.0, 1.0)

def test_psnr():
	clean = utils.LoadImage('test_files/bridge.png')
	noisy = _add_noise(clean, sigma=0.15)
	assert pytest.approx(utils.PSNR(clean, noisy), 0.01) == 17.067
	clean = utils.LoadImage('test_files/canal.png')
	noisy = _add_noise(clean, sigma=0.15)
	assert pytest.approx(utils.PSNR(clean, noisy), 0.01) == 16.963

def test_estimate_sigma():
	clean = utils.LoadImage('test_files/bridge.png')
	noisy = _add_noise(clean, sigma=0.15)
	noise = float(jnp.mean(utils.EstimateNoiseSigma(noisy)))
	assert pytest.approx(noise, abs=0.02) == 0.15
	clean = utils.LoadImage('test_files/canal.png')
	noisy = _add_noise(clean, sigma=0.15)
	noise = float(jnp.mean(utils.EstimateNoiseSigma(noisy)))
	assert pytest.approx(noise, abs=0.02) == 0.15

def test_nlmeans():
	clean = utils.LoadImage('test_files/canal.png')
	noisy = _add_noise(clean, sigma=0.15)
	denoised = nlmeans.nlmeans(img=noisy, **nlmeans.default_nlmeans_params(noisy))
	utils.SaveImage(denoised, 'test_out/nlmeans_denoised.png')
	compare = utils.MergeSideBySide(noisy, denoised)
	utils.SaveImage(compare, 'test_out/nlmeans_compare.png')
	psnr = float(utils.PSNR(clean, denoised))
	print(f'PSNR: {psnr}')
	assert psnr > 27.0

# def test_nlmeans_patchwise():
# 	clean = utils.LoadImage('test_files/canal.png')
# 	noisy = _add_noise(clean, sigma=0.15)
# 	denoised = nlmeans.nlmeans_patchwise(img=noisy, search_range=21, patch_size=3)
# 	utils.SaveImage(denoised, 'test_out/nlmeans_denoised_patchwise.png')
# 	compare = utils.MergeSideBySide(noisy, denoised)
# 	utils.SaveImage(compare, 'test_out/nlmeans_compare_patchwise.png')
# 	psnr = float(utils.PSNR(clean, denoised))
# 	print(f'PSNR: {psnr}')
# 	assert psnr > 27.0