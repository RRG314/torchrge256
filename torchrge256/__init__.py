"""
TorchRGE256: PyTorch-Native RGE-256 PRNG

A PyTorch-native implementation of the RGE-256 pseudorandom number generator.
RGE-256 is a 256-bit ARX (Add-Rotate-XOR) generator whose rotation schedule
is derived from geometric entropy constants obtained through Recursive Division
Tree (RDT) analysis.

Author: Steven Reid
ORCID: 0009-0003-9132-3410
"""

from .torchrge256 import TorchRGE256

__version__ = "0.1.1"
__all__ = ["TorchRGE256"]
