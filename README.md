# TorchRGE256
A PyTorch-native implementation of the RGE-256 pseudorandom number generator.

Author: Steven Reid  
ORCID: 0009-0003-9132-3410  
Zenodo Preprint: https://zenodo.org/records/17713219  
Repository: https://github.com/RRG314/torchrge256

## Overview
TorchRGE256 is a PyTorch-native implementation of the RGE-256 pseudorandom number generator (PRNG). RGE-256 is a 256-bit ARX (Add-Rotate-XOR) generator whose rotation schedule is derived from geometric entropy constants obtained through Recursive Division Tree (RDT) analysis. The PyTorch version is designed for machine learning workflows, GPU execution, reproducible training, and deterministic data generation.

TorchRGE256 provides:
- A 256-bit internal state (8 x 32-bit words)
- Deterministic and reproducible output
- CPU and CUDA support
- Full state checkpointing
- Domain separation for independent streams
- High-level random sampling utilities (uniform, normal, randint, Bernoulli, permutation, shuffle, dropout masks, and more)

This implementation is written entirely in Python and uses only PyTorch and the Python standard library.

## Scientific Background
The rotation constants in RGE-256 are derived from three geometric entropy values that emerge from Recursive Division Tree (RDT) analysis:
- zeta_1 ≈ 1.585
- zeta_2 ≈ 1.926
- zeta_3 ≈ 1.262

These constants represent stable entropy ratios observed in computational experitments with entropy.

## Key Features
- PyTorch-native PRNG
- Reproducible training
- CUDA support
- Domain separation
- Deterministic ARX core

## Installation
Once published:
pip install torchrge256

## Example
import torch
from torchrge256 import TorchRGE256
rng = TorchRGE256(seed=123)
x = rng.rand((3, 3))

## Citation
@misc{reid2025rge256,
  author = {Reid, Steven},
  title = {RGE-256: A New ARX-Based Pseudorandom Number Generator With Structured Entropy and Empirical Validation},
  year = {2025},
  doi = {10.5281/zenodo.17713219}
}

## License
MIT License