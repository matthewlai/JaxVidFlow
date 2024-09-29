import sys

import numpy as np
import pytest

sys.path.append('.')

from JaxVidFlow import nlmeans, utils

def test_psnr():
	clean = utils.LoadImage('test_files/bridge.png')
	noisy = utils.LoadImage('test_files/bridge_noisy.png')
	denoised = utils.LoadImage('test_files/bridge_denoised.png')
	assert pytest.approx(nlmeans.psnr(clean, noisy), 0.01) == 28.843
	assert pytest.approx(nlmeans.psnr(clean, denoised), 0.01) == 29.141

@pytest.mark.skip(reason="This is really slow (as designed). We saved the image to compare against instead.")
def test_naive_nlmeans():
	clean = utils.LoadImage('test_files/bridge.png')
	noisy = utils.LoadImage('test_files/bridge_noisy.png')
	denoised = nlmeans.naive_nlmeans(img=noisy, strength=0.1, search_range=3, patch_size=7)
	utils.SaveImage(denoised, 'test_out/bridge_denoised.png')
	psnr = float(nlmeans.psnr(clean, denoised))
	assert psnr > 28.0

def test_make_offsets():
	offsets = nlmeans._make_offsets(7)
	set_map = np.zeros((7, 7), dtype=np.uint8)
	for row in range(offsets.shape[0]):
		set_map[tuple(offsets[row] + 3)] = 1
	expected = np.array(
		[[1, 1, 1, 1, 1, 1, 1],
 		 [1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
	assert (set_map == expected).all()

def test_fast_nlmeans():
	clean = utils.LoadImage('test_files/bridge.png')
	noisy = utils.LoadImage('test_files/bridge_noisy.png')
	denoised = nlmeans.nlmeans(img=noisy, strength=0.1, search_range=3, patch_size=7)
	utils.SaveImage(denoised, 'test_out/bridge_denoised.png')
	compare = np.copy(noisy)
	compare[:, :256] = np.array(denoised[:, :256])
	utils.SaveImage(compare, 'test_out/nlmeans_compare.png')
	psnr = float(nlmeans.psnr(clean, denoised))
	print(f'PSNR: {psnr}')