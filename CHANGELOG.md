# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-12-09

### Added
- Counter-mode implementation of RGE-256 (`TorchRGE256ctr`, `RGE256ctr_Torch`)
  - Simpler ARX structure with 64-bit counter
  - Optimized for batch generation (8 x 32-bit values per block)
  - Full API compatibility with standard `TorchRGE256`
  - Complete state management (save/restore/fork)
- Comprehensive test suite for counter-mode (`TEST_CTR_MODE.py`)
- Release notes document (`RELEASE_NOTES_v1.1.0.md`)

### Changed
- Bumped version to 1.1.0 in `pyproject.toml` and `__init__.py`

## [0.1.1] - 2025-11-26

### Added
- GitHub Actions workflows for automated PyPI publishing
  - `ci.yml`: Continuous integration testing on PRs and pushes
  - `publish-pypi.yml`: Automated publishing to production PyPI on releases
  - `publish-test-pypi.yml`: Automated publishing to Test PyPI on test tags
- Comprehensive publishing documentation in `PUBLISHING.md`
- `CHANGELOG.md` for tracking version history
- `.gitignore` for Python/PyPI artifacts

### Changed
- Restructured package for proper PyPI distribution
  - Moved implementation from `src` to `torchrge256/torchrge256.py`
  - Created proper package directory structure
  - Removed placeholder files
- Updated `pyproject.toml`:
  - Bumped version to 0.1.1
  - Added comprehensive metadata and classifiers
  - Fixed license format to use SPDX expression
  - Added keywords and project URLs
- Added `MANIFEST.in` for proper file inclusion in distributions

### Fixed
- State save/restore test logic in `torchrge256-test.ipynb`
  - Fixed test to save state before generation instead of after
  - Test now correctly verifies state save/restore functionality

## [0.1.0] - 2025-01-XX

### Added
- Initial release of TorchRGE256
- PyTorch-native implementation of RGE-256 PRNG
- 256-bit internal state (8 x 32-bit words)
- Full state checkpointing (state_dict/load_state_dict)
- CPU and CUDA support
- High-level random sampling utilities:
  - `rand()`, `randn()`, `randint()`
  - `uniform()`, `normal()`, `exponential()`
  - `bernoulli()`, `dropout_mask()`
  - `shuffle()`, `permutation()`, `choice()`
- Domain separation for independent streams via `fork()`
- Rotation constants derived from geometric entropy values
- MIT License
- Comprehensive test suite in Jupyter notebook

[1.1.0]: https://github.com/RRG314/torchrge256/compare/v0.1.1...v1.1.0
[0.1.1]: https://github.com/RRG314/torchrge256/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/RRG314/torchrge256/releases/tag/v0.1.0
