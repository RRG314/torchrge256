# TorchRGE256

[![PyPI version](https://badge.fury.io/py/torchrge256.svg)](https://badge.fury.io/py/torchrge256)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/torchrge256)](https://pepy.tech/project/torchrge256)
[![CI](https://github.com/RRG314/torchrge256/actions/workflows/ci.yml/badge.svg)](https://github.com/RRG314/torchrge256/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17713219.svg)](https://doi.org/10.5281/zenodo.17713219)

A PyTorch-native implementation of the RGE-256 pseudorandom number generator.

**Author:** Steven Reid
**ORCID:** [0009-0003-9132-3410](https://orcid.org/0009-0003-9132-3410)
**Zenodo Preprint:** https://zenodo.org/records/17713219
**Repository:** https://github.com/RRG314/torchrge256

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

```bash
pip install torchrge256
```

Or install a specific version:
```bash
pip install torchrge256==1.1.0
```

## Quick Start

```python
import torch
from torchrge256 import TorchRGE256

# Create generator with seed
rng = TorchRGE256(seed=123)

# Generate random tensors
x = rng.rand((3, 3))        # Uniform [0, 1)
y = rng.randn((5, 5))       # Normal distribution
z = rng.randint(0, 10, (4,)) # Random integers

# Counter-mode variant (v1.1.0+)
from torchrge256 import RGE256ctr_Torch
rng_ctr = RGE256ctr_Torch(seed=42, device='cuda')
x = rng_ctr.rand((100, 100))
```

## Citation
@misc{reid2025rge256,
  author = {Reid, Steven},
  title = {RGE-256: A New ARX-Based Pseudorandom Number Generator With Structured Entropy and Empirical Validation},
  year = {2025},
  doi = {10.5281/zenodo.17713219}
}

## License
MIT License