# Experimental Environment

## Hardware
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- GPU Memory: 6 GB
- NVIDIA Driver: 595.99.02

## Software
- Operating System: Ubuntu 26.04 LTS
- Kernel: 7.0.0-38-generic
- Python: 3.11.13
- PyTorch: 2.5.1+cu124
- Torchvision: 0.20.1+cu124
- NumPy: 2.4.6
- PyTorch CUDA Runtime: 12.4
- CUDA availability in PyTorch: True

## Notes
The project was executed using a Python virtual environment.
The original repository documentation specifies Python 3.9.13,
PyTorch 1.12.1, torchvision 0.13.1, and CUDA 11.3.
The experiments reported in this project used the versions listed above,
so the software environment differs from the originally documented setup.

The NVIDIA driver reports CUDA 13.2 support, while PyTorch uses its
bundled CUDA 12.4 runtime. These version numbers refer to different
components and are not necessarily contradictory.
