# Wan2.1 Text-to-Video Model

This repository contains the Wan2.1 text-to-video model, adapted for macOS with M1 Pro chip. This adaptation allows macOS users to run the model efficiently, overcoming CUDA-specific limitations.

## Introduction

The Wan2.1 model is an open-source text-to-video generation model. It transforms textual descriptions into video sequences, leveraging advanced machine learning techniques.

## Changes for macOS

This version includes modifications to make the model compatible with macOS, specifically for systems using the M1 Pro chip. Key changes include:

- Adaptation of CUDA-specific code to work with MPS (Metal Performance Shaders) on macOS.
- Environment variable settings for MPS fallback to CPU for unsupported operations.
- Adjustments to command-line arguments for better compatibility with macOS.

## Installation Instructions

Follow these steps to set up the environment on macOS:

1. **Install Homebrew**: If not already installed, use Homebrew to manage packages.
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python 3.10+**:
   ```bash
   brew install python@3.10
   ```

3. **Create and Activate a Virtual Environment**:
   ```bash
   python3.10 -m venv venv_wan
   source venv_wan/bin/activate
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install einops
   ```

## Usage

To generate a video, use the following command:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python generate.py --task t2v-1.3B --size "480*832" --frame_num 16 --sample_steps 25 --ckpt_dir ./Wan2.1-T2V-1.3B --offload_model True --t5_cpu --device mps --prompt "Lion running under snow in Samarkand" --save_file output_video.mp4
```

## Optimization Tips

- **Use CPU for Large Models**: If you encounter memory issues, use `--device cpu`.
- **Reduce Resolution and Frame Count**: Use smaller resolutions and fewer frames to reduce memory usage.
- **Monitor System Resources**: Keep an eye on memory usage and adjust parameters as needed.

## Acknowledgments

This project is based on the original Wan2.1 model. Special thanks to the original authors and contributors for their work.
