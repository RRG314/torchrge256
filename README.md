
# TorchRGE256: A Deterministic 256-bit PRNG for PyTorch (CPU and CUDA)

**Author:** Steven Reid
**ORCID:** 0009-0003-9132-3410
**Repository:** [https://github.com/RRG314/torchrge256](https://github.com/RRG314/torchrge256)
**Zenodo Preprint:** (RGE-256: A New ARX-Based Pseudorandom Number Generator With Structured Entropy and Empirical Validation):** [https://zenodo.org/records/17713219](https://zenodo.org/records/17713219)

TorchRGE256 is a deterministic 256-bit pseudorandom number generator implemented as a PyTorch-compatible module. It provides stable and reproducible randomness for both CPU and CUDA devices and includes full support for distribution sampling, state checkpointing, domain separation, and independent substreams. TorchRGE256 operates independently from PyTorch's global RNG, enabling reproducible and isolated randomness streams for machine learning workflows.

This implementation is intended for scientific computing, simulation, deterministic machine learning experiments, and other settings where reproducibility and controlled randomness are required.

## Features

* 256-bit ARX-based pseudorandom number generator
* Deterministic behavior on CPU and CUDA
* Complete distribution API including uniform, normal, integer, Bernoulli, exponential, dropout masks, and others
* Reproducible and exportable internal state
* Independent substreams through domain separation
* Manual seeding and device control
* No external dependencies beyond PyTorch
* Suitable for training, reinforcement learning, simulation, stochastic modeling, augmentation, and debugging

## Installation

To clone the repository:

```
git clone https://github.com/RRG314/torchrge256
cd torchrge256
```

 PyPI installation:

```
pip install torchrge256
```

## Basic Usage

```python
from torchrge256 import TorchRGE256
import torch

rng = TorchRGE256(seed=42)

x = rng.rand((3, 3))
y = rng.randn((4, 4))

print(x)
print(y)
```

## GPU Usage

```python
rng = TorchRGE256(seed=42, device="cuda")
x = rng.rand((3, 3))
print(x.device)   # cuda:0
```

## Reproducibility

TorchRGE256 supports exact reproducibility through complete state checkpointing.

```python
rng = TorchRGE256(seed=123, device="cuda")

state = rng.state_dict()
x1 = rng.rand((1000,))

rng.load_state_dict(state)
x2 = rng.rand((1000,))

print(torch.equal(x1, x2))   # True
```

This ensures that experiments can be repeated exactly, which is essential for scientific research, training reproducibility, debugging, and evaluation.

## Distribution API

### Uniform Distribution [0, 1)

```python
rng.rand((m, n))
```

### Normal (Gaussian) Distribution

```python
rng.randn((m, n))
rng.normal(mean, std, (m, n))
```

### Integer Sampling

```python
rng.randint(0, 10, (m, n))
```

### Bernoulli Distribution

```python
rng.bernoulli(0.3, (m, n))
```

### Dropout Mask (Scaled)

```python
rng.dropout_mask((m, n), p=0.5)
```

### Uniform Range

```python
rng.uniform(2.0, 5.0, (m, n))
```

### Exponential Distribution

```python
rng.exponential(rate=1.5, shape=(m, n))
```

### Permutations

```python
rng.permutation(10)
```

### Sampling With or Without Replacement

```python
rng.choice(tensor, 5, replace=False)
```

## Substreams and Domain Separation

Independent substreams can be created using the `fork` method. Each substream maintains a unique internal state and domain, ensuring that random sequences do not overlap.

```python
main_rng = TorchRGE256(seed=999)

layer_rng = main_rng.fork("layer")
augment_rng = main_rng.fork("augment")

print(layer_rng.rand((5,)))
print(augment_rng.rand((5,)))
```

## Device Control

```python
rng.cpu()
rng.cuda()
rng.to("cuda:1")
```

Generated tensors are placed directly on the target device. The internal PRNG state remains CPU-based, similar to PyTorch's default generators.

## Internal Design Summary

TorchRGE256 is based on a 256-bit ARX core structure using:

* add, rotate, xor operations
* a rotation schedule derived from the zeta parameters
* a key schedule mixed from domain-extended seed material
* SHA-512 based domain hashing
* warmup rounds for initial state dispersion

The design emphasizes determinism, reproducibility, isolation of randomness streams, and compatibility with PyTorch tensor workflows.

## Security Notice

TorchRGE256 is not a cryptographically secure random number generator.
It must not be used for encryption, key generation, gambling, lottery systems, blockchain consensus, authentication tokens, or any security-dependent purpose.
It is intended strictly for machine learning, simulation, and scientific computing.

## Citation

If TorchRGE256 or RGE-256 is used in academic work, please cite:

Reid, Steven. “RGE-256: A 256-bit ARX-Based Random Number Generator.”
Zenodo (2025). [https://zenodo.org/records/17713219](https://zenodo.org/records/17713219)
ORCID: 0009-0003-9132-3410

## License

TorchRGE256 is released under the MIT license.

## Contributing

Feature requests, bug reports, and collaboration inquiries are welcome through the GitHub repository:

[https://github.com/RRG314/torchrge256](https://github.com/RRG314/torchrge256)

