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

# Install JAX with the experimental METAL backend.
pip3 install jax-metal
```

### Common
```
# Install other dependencies.
pip3 install tqdm
```
