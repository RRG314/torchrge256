# Publishing to PyPI

## Package Structure
The package is properly structured for PyPI publishing:
```
torchrge256/
├── torchrge256/           # Package directory
│   ├── __init__.py       # Package initialization
│   └── torchrge256.py    # Main implementation
├── pyproject.toml        # Build configuration
├── MANIFEST.in           # Additional files to include
├── README.md             # Package description
└── LICENSE               # MIT License
```

## Version
Current version: **0.1.1**

Update version in two places:
1. `pyproject.toml` - line 7: `version = "0.1.1"`
2. `torchrge256/__init__.py` - line 15: `__version__ = "0.1.1"`

## Build the Package

```bash
# Install build tools
pip install build twine

# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build source distribution and wheel
python -m build
```

This creates:
- `dist/torchrge256-0.1.1.tar.gz` (source distribution)
- `dist/torchrge256-0.1.1-py3-none-any.whl` (wheel)

## Validate the Package

```bash
# Check package (note: twine may show warnings about metadata 2.4 format, this is expected)
twine check dist/*

# Test install locally
pip install dist/torchrge256-0.1.1-py3-none-any.whl
```

## Publish to PyPI

### Test PyPI (recommended first)
```bash
# Upload to Test PyPI
twine upload --repository testpypi dist/*

# Test install from Test PyPI
pip install --index-url https://test.pypi.org/simple/ torchrge256
```

### Production PyPI
```bash
# Upload to PyPI
twine upload dist/*
```

You'll need PyPI credentials. Create a `.pypirc` file:
```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-...

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-...
```

## After Publishing

Users can install with:
```bash
pip install torchrge256
```

## Notes
- The twine check warning about 'license-expression' and 'license-file' fields is due to setuptools using the newer metadata format (2.4). PyPI accepts this format correctly.
- Always test on Test PyPI before publishing to production PyPI
- PyPI releases are permanent and cannot be deleted (only yanked)
