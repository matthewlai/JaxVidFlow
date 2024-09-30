# JaxVidFlow
Video processing pipeline using JAX. Especially suitable for experimenting with custom video processing operations.

## Why?
* FFmpeg has done great things for the community over the past decades, and has been the go-to for automated video processing, but -
  * It's really hard to extend with filters. Writing high performance C code is hard, and very hardware-dependent
  * Only a handful of filters have GPU implementations, and generally using them requires very messy command line options
  * Every new CPU or GPU architecture requires new code
  * Floating point formats not supported, requiring carefully choosing pixel formats to minimize quality loss for multi-stage processing, and different filters support different formats.
* [JAX](https://jax.readthedocs.io/en/latest/index.html) is a high performance and very user-friendly array computing library.
  * Write simple Numpy expressions, get well-optimised native code performance on CPU/GPU/TPU
  * Very easy to implement most custom operations. As long as you can express it as matrix operations, you are 80% of the way there!
  * Code generation for:
    * CPUs (compiled into Eigen operations with good vectorization for x86, ARM, and other architectures)
    * GPU (NVIDIA CUDA is best supported, AMD ROCm experimental, Intel oneAPI also experimental, Apple Metal Performance Shaders Graph on all Apple GPUs also experimental)
    * Google TPUs
    * Future architectures as they come out, without having to change our code (in theory)
  * We can do everything in floating point. FP is the state of the art for minimal-loss multi-stage video processing, and we can do it very fast with GPUs (and CPUs with SIMD).

## Current Status

You can run benchmarks.py to see how fast things are, but it doesn't really have a UI yet (not even a CLI). It's just a library. See examples/process_dive_video.py for a typical pipeline setup for filtering a video.

### Implemented functions
* Decode / encode pipeline using FFmpeg (through PyAV)
  * Reasonably optimised - only about 10% slower than using FFmpeg directly for a straight transcode with hardware encoding
    * If you just want to do a straight transcode, I would recommend using FFmpeg instead.
  * Supports hardware encoders (hardware decoders are not supported due to PyAV limitation, but software decoders are very fast anyways)

### Transforms
* YUV to RGB and back, including chroma subsampling/supersampling
* Rec709 to linear and back
* 3D LUT application with trilinear filtering
* Resizing with Lanczos interpolation (ok, this is really just a one line call to jax.image.resize())
* Denoising using NL-Means (both pixelwise and blockwise variants implemented)
  * ~50 fps at 4K on my NVIDIA RTX 3060 Ti, compared to ~2 fps with FFmpeg's CPU implementation (I wasn't able to get the OpenCL version to work)

## Installation Instructions
### Linux (Debian/Ubuntu) or Windows (with WSL2, in an Ubuntu VM)
```
# Install dependencies.
sudo apt install pkg-config python3 libavformat-dev libavcodec-dev libavdevice-dev \
  libavutil-dev libswscale-dev libswresample-dev libavfilter-dev

# Setup a virtual environment (optional).
python3 -m venv venv

# Activate the virtual environment (optional).
source venv/bin/activate

# Install PyAV from source (this links against system ffmpeg libraries, which is necessary to get hardware-accelerated encoders).
pip3 install av --no-binary av

# Install JAX (with NVIDIA GPU support).
pip3 install jax[cuda12]

# Or, install CPU-only JAX.
pip3 install jax

# See Jax documentation for installing JAX with experimental backends (eg. AMD ROCm):
# https://jax.readthedocs.io/en/latest/installation.html
```

### Mac (with experimental JAX METAL support)
```
# Install homebrew (https://brew.sh/) and Python
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python3

# Install ffmpeg.
brew install ffmpeg pkg-config

# Setup a virtual environment (optional).
python3 -m venv venv

# Activate the virtual environment (optional).
source venv/bin/activate

# Install PyAV from source (this links against system ffmpeg libraries, which is necessary to get hardware-accelerated encoders).
pip3 install av --no-binary av

# Install JAX with the experimental METAL backend.
pip3 install jax-metal
```

### Common
```
# Install other dependencies.
pip3 install tqdm pytest
```

## Acknowledgements
* The included [DJI LUT](https://github.com/matthewlai/JaxVidFlow/blob/main/luts/D_LOG_M_to_Rec_709_LUT_ZG_Rev1.cube) is a [better DJI D-Log-M to Linear LUT](https://www.zebgardner.com/photo-and-video-editing/dji-d-log-m-colorgrading) created by [Zeb Gadner](https://www.zebgardner.com/), included here with his permission.
* The canal.png test image is by [Hervé BRY on Flickr](https://www.flickr.com/photos/setaou/2162752903/), licensed under Attribution-NonCommercial-ShareAlike (CC BY-NC-SA 2.0). Cropped from original.
