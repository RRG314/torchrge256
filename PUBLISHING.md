# Publishing to PyPI

## Package Structure
The package is properly structured for PyPI publishing:
```
torchrge256/
├── .github/workflows/     # GitHub Actions workflows
│   ├── ci.yml            # CI testing (builds on PR/push)
│   ├── publish-pypi.yml  # Production PyPI publishing
│   └── publish-test-pypi.yml  # Test PyPI publishing
├── torchrge256/          # Package directory
│   ├── __init__.py       # Package initialization
│   └── torchrge256.py    # Main implementation
├── pyproject.toml        # Build configuration
├── MANIFEST.in           # Additional files to include
├── README.md             # Package description
└── LICENSE               # MIT License
```

## Version Management

Current version: **0.1.1**

Update version in two places before publishing:
1. `pyproject.toml` - line 7: `version = "0.1.1"`
2. `torchrge256/__init__.py` - line 15: `__version__ = "0.1.1"`

---

## Method 1: Automated Publishing (Recommended)

### Setup PyPI Trusted Publishers (One-time setup)

**For Test PyPI:**
1. Go to https://test.pypi.org/manage/account/publishing/
2. Add a new publisher:
   - **PyPI Project Name**: `torchrge256`
   - **Owner**: `RRG314` (your GitHub username/org)
   - **Repository name**: `torchrge256`
   - **Workflow name**: `publish-test-pypi.yml`
   - **Environment name**: `testpypi`

**For Production PyPI:**
1. Go to https://pypi.org/manage/account/publishing/
2. Add a new publisher:
   - **PyPI Project Name**: `torchrge256`
   - **Owner**: `RRG314`
   - **Repository name**: `torchrge256`
   - **Workflow name**: `publish-pypi.yml`
   - **Environment name**: `pypi`

### Setup GitHub Environments

1. Go to your GitHub repo: Settings → Environments
2. Create environment: `testpypi`
3. Create environment: `pypi` (add protection rules like requiring approvals)

### Publishing to Test PyPI

**Option A: Create a test tag**
```bash
git tag v0.1.1-test1
git push origin v0.1.1-test1
```

**Option B: Manual trigger**
1. Go to Actions tab in GitHub
2. Select "Publish to Test PyPI" workflow
3. Click "Run workflow"

### Publishing to Production PyPI

**Create a GitHub Release:**
1. Go to your repo → Releases → "Draft a new release"
2. Create a new tag: `v0.1.1`
3. Title: `v0.1.1` or `Release 0.1.1`
4. Description: Write release notes
5. Click "Publish release"

The workflow will automatically:
- Build the package
- Publish to PyPI
- Make it available at `pip install torchrge256`

---

## Method 2: Manual Publishing (Fallback)

### Build the Package

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

### Validate the Package

```bash
# Check package
twine check dist/*

# Test install locally
pip install dist/torchrge256-0.1.1-py3-none-any.whl
```

### Publish to Test PyPI

```bash
# Upload to Test PyPI
twine upload --repository testpypi dist/*

# Test install from Test PyPI
pip install --index-url https://test.pypi.org/simple/ torchrge256
```

### Publish to Production PyPI

```bash
# Upload to PyPI
twine upload dist/*
```

### Manual Publishing with API Tokens

If not using Trusted Publishers, create a `.pypirc` file:
```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmc...

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-AgENdGVzdC5weXBpLm9yZw...
```

Get API tokens from:
- PyPI: https://pypi.org/manage/account/token/
- Test PyPI: https://test.pypi.org/manage/account/token/

---

## CI/CD Workflows

### CI (Continuous Integration)
- **Trigger**: Push/PR to main/master branch
- **Actions**: Build package, run twine check, test install
- **Purpose**: Ensure package builds correctly before merging

### Publish to Test PyPI
- **Trigger**: Tags matching `v*-test*` or manual trigger
- **Actions**: Build and publish to Test PyPI
- **Purpose**: Test releases before production

### Publish to PyPI
- **Trigger**: GitHub Release published or manual trigger
- **Actions**: Build and publish to Production PyPI
- **Purpose**: Production releases

---

## Release Checklist

- [ ] Update version in `pyproject.toml`
- [ ] Update version in `torchrge256/__init__.py`
- [ ] Update `CHANGELOG.md` (if exists)
- [ ] Test locally: `python -m build && pip install dist/*.whl`
- [ ] Commit version bump: `git commit -am "Bump version to X.Y.Z"`
- [ ] Push to main: `git push`
- [ ] (Optional) Test on Test PyPI with tag: `git tag vX.Y.Z-test1 && git push origin vX.Y.Z-test1`
- [ ] Create GitHub Release with tag `vX.Y.Z`
- [ ] Verify publish succeeds in Actions tab
- [ ] Test install: `pip install torchrge256`

---

## After Publishing

Users can install with:
```bash
pip install torchrge256
```

Or upgrade:
```bash
pip install --upgrade torchrge256
```

---

## Notes

- **Trusted Publishers** (OIDC) is more secure than API tokens - no credentials stored
- The CI workflow runs on every PR to catch build issues early
- Always test on Test PyPI before publishing to production PyPI
- PyPI releases are permanent and cannot be deleted (only yanked)
- The twine check warning about 'license-expression' fields is expected (metadata 2.4 format)
- GitHub Actions will show detailed logs if publishing fails
