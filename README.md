# JaxVidFlow
Experimental video processing pipeline using Jax.

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
* The included [DJI LUT](https://github.com/matthewlai/JaxVidFlow/blob/main/luts/D_LOG_M_to_Rec_709_LUT_ZG_Rev1.cube) is a better DJI D-Log-M to Linear LUT [created](https://www.zebgardner.com/photo-and-video-editing/dji-d-log-m-colorgrading) by [Zeb Gadner](https://www.zebgardner.com/), included here with his permission.
* The canal.png test image is by [Hervé BRY on Flickr](https://www.flickr.com/photos/setaou/2162752903/), licensed under Attribution-NonCommercial-ShareAlike (CC BY-NC-SA 2.0), Cropped.
