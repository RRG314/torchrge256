# Release Notes: v0.1.2 - Counter-Mode Support

## What's New

### New Feature: RGE-256 Counter-Mode Implementation

Added a counter-mode variant of RGE-256 (`TorchRGE256ctr`) that provides an alternative architecture optimized for batch generation:

- **Simpler ARX structure**: Uses 64-bit counter with streamlined mixing
- **Batch generation**: Produces 8 x 32-bit values per block
- **Full API compatibility**: Same interface as `TorchRGE256`
- **State management**: Complete save/restore support

### Usage

```python
from torchrge256 import RGE256ctr_Torch, TorchRGE256ctr

# Create counter-mode generator
rng = RGE256ctr_Torch(seed=42, device='cuda')

# Generate random tensors
x = rng.rand((100, 100))        # Uniform [0, 1)
y = rng.randn((50, 50))         # Standard normal  
z = rng.randint(0, 10, (20,))   # Integers [0, 10)

# State management
state = rng.get_state()
rng2 = RGE256ctr_Torch.from_state(state)
```

### API Methods

Both class names work identically:
- `RGE256ctr_Torch` - Primary class name
- `TorchRGE256ctr` - Alias for consistency

Methods:
- `rand(shape, dtype)` - Uniform random [0, 1)
- `randn(shape, dtype)` - Normal distribution
- `randint(low, high, shape, dtype)` - Random integers
- `random_uint32(n)` - Raw 32-bit integers
- `get_state()` / `set_state()` / `from_state()` - State management
- `manual_seed(seed)` - Reseed generator
- `to(device)` / `cuda()` / `cpu()` - Device movement

## Files Changed

- `torchrge256/torchrge256ctr.py` - New counter-mode implementation
- `torchrge256/__init__.py` - Export new classes
- `TEST_CTR_MODE.py` - Comprehensive test suite
- `pyproject.toml` - Version bump to 0.1.2
- `CHANGELOG.md` - Updated (if exists)

## Package Validation

✅ **Build Status**: Successfully built
✅ **Distribution**: Both wheel and source distribution created
✅ **Contents**: All files included correctly
✅ **Metadata**: Package metadata validated

## Build Artifacts

- `dist/torchrge256-0.1.2-py3-none-any.whl` (9.5 KB)
- `dist/torchrge256-0.1.2.tar.gz` (9.2 KB)

## Release Process

The package is ready for automated PyPI publishing:

1. **Merge this branch to main** (or merge directly)
2. **Create GitHub Release**:
   - Go to: https://github.com/RRG314/torchrge256/releases/new
   - Tag: `v0.1.2`
   - Title: `Release 0.1.2 - Counter-Mode Support`
   - Description: Add these release notes
   - Publish release

3. **Automated Publishing**: GitHub Actions will automatically:
   - Build the package
   - Publish to PyPI
   - Make it available via `pip install torchrge256==0.1.2`

## Backwards Compatibility

✅ Fully backwards compatible - existing `TorchRGE256` class unchanged
✅ New feature only - no breaking changes
✅ All existing code continues to work

## What's Next

After publishing:
- Test install: `pip install --upgrade torchrge256`
- Verify: `python -c "from torchrge256 import TorchRGE256ctr; print('OK')"`
- Documentation updates (if needed)
- Announce on relevant channels

---

**Author**: Steven Reid  
**ORCID**: 0009-0003-9132-3410  
**Date**: December 9, 2025
