# Macenko PCA

[![Build](https://github.com/LavLabInfrastructure/macenko-pca/actions/workflows/build.yml/badge.svg)](https://github.com/LavLabInfrastructure/macenko-pca/actions/workflows/build.yml)
[![Tests](https://github.com/LavLabInfrastructure/macenko-pca/actions/workflows/pytest.yml/badge.svg)](https://github.com/LavLabInfrastructure/macenko-pca/actions/workflows/pytest.yml)
[![Lint](https://github.com/LavLabInfrastructure/macenko-pca/actions/workflows/lint.yml/badge.svg)](https://github.com/LavLabInfrastructure/macenko-pca/actions/workflows/lint.yml)
[![PyPI - Version](https://img.shields.io/pypi/v/macenko-pca.svg)](https://pypi.org/project/macenko-pca)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/macenko-pca.svg)](https://pypi.org/project/macenko-pca)

-----

High-performance stain matrix estimation and colour deconvolution for histology images using the **Macenko PCA** method, with the compute-intensive core written in **Rust** via [PyO3](https://pyo3.rs/).

This implements the method described in:

> Macenko, M. et al. "A method for normalizing histology slides for quantitative analysis." *ISBI 2009*.

## Features

| Feature | Detail |
|---------|--------|
| **Performance** | Core SVD, PCA projection, and angle binning run in compiled Rust with [Rayon](https://github.com/rayon-rs/rayon) parallelism |
| **Simple API** | Six functions cover the full workflow — estimate stain vectors, decompose, and reconstruct |
| **NumPy native** | Accepts and returns standard NumPy arrays — no special data structures required |
| **Precision-aware** | Pass `float32` arrays to halve RAM usage — the dtype of your input controls which Rust code path runs |
| **Cross-platform wheels** | Built with [maturin](https://github.com/PyO3/maturin) for Linux, macOS (x86_64 + arm64), and Windows |

## Quick Start

### Installation

```console
pip install macenko-pca
```

### Full Workflow Example

```python
import numpy as np
from macenko_pca import (
    rgb_separate_stains_macenko_pca,
    rgb_color_deconvolution,
    reconstruct_rgb,
)

# Load or create an RGB image (H×W×3, values in [0, 255])
im_rgb = np.random.rand(256, 256, 3) * 255.0

# 1. Estimate the 3×3 stain matrix from the image
stain_matrix = rgb_separate_stains_macenko_pca(im_rgb)
print("Stain matrix:\n", stain_matrix)

# 2. Decompose the image into per-stain concentration channels
concentrations = rgb_color_deconvolution(im_rgb, stain_matrix)

hematoxylin = concentrations[:, :, 0]
eosin = concentrations[:, :, 1]
residual = concentrations[:, :, 2]

# 3. Modify concentrations (e.g. isolate hematoxylin only)
concentrations_h_only = concentrations.copy()
concentrations_h_only[:, :, 1] = 0.0  # zero-out eosin
concentrations_h_only[:, :, 2] = 0.0  # zero-out residual

# 4. Reconstruct back to RGB
im_hematoxylin_only = reconstruct_rgb(concentrations_h_only, stain_matrix)
```

### Step-by-Step API

```python
from macenko_pca import (
    rgb_to_sda,
    separate_stains_macenko_pca,
    color_deconvolution,
    reconstruct_rgb,
)

# Convert RGB to SDA (Stain Density Absorbance) space
im_sda = rgb_to_sda(im_rgb)

# Estimate stain vectors from the SDA image
stain_matrix = separate_stains_macenko_pca(im_sda)

# Decompose SDA image into stain concentrations
concentrations = color_deconvolution(im_sda, stain_matrix)

# Reconstruct RGB from (possibly modified) concentrations
im_reconstructed = reconstruct_rgb(concentrations, stain_matrix)
```

### Half the RAM — Use float32

```python
# Simply cast your input to float32 — the Rust backend will use f32 throughout
im_rgb_f32 = im_rgb.astype(np.float32)
stain_matrix = rgb_separate_stains_macenko_pca(im_rgb_f32)       # f32
concentrations = rgb_color_deconvolution(im_rgb_f32, stain_matrix)  # f32
reconstructed = reconstruct_rgb(concentrations, stain_matrix)        # f32
```

### Precision / dtype rules

The dtype of your input array controls which Rust code path is taken:

| Input dtype | Computation dtype | Notes |
|-------------|-------------------|-------|
| `float64`   | f64               | Full precision (default for plain Python floats) |
| `float32`   | f32               | ≈ half the RAM — recommended when full precision is unnecessary |
| `float16`   | f32               | Promoted to f32 (no f16 LAPACK exists) |
| integer types | f64             | Promoted to f64 for backward compatibility |

The return array's dtype always matches the computation dtype.

## API Reference

### Stain Matrix Estimation

#### `rgb_separate_stains_macenko_pca(im_rgb, ...)`

End-to-end: takes an RGB image `(H, W, 3)` and returns a `(3, 3)` stain matrix.

#### `separate_stains_macenko_pca(im_sda, ...)`

Lower-level: operates on an image already in SDA space.

### Colour Conversion

#### `rgb_to_sda(im_rgb, ...)`

Converts an RGB image or matrix to SDA (stain-density-absorbance) space.

### Colour Deconvolution (Applying Stain Vectors)

#### `color_deconvolution(im_sda, stain_matrix)`

Decomposes an SDA image into per-stain concentration channels using the inverse of the stain matrix. Each output channel *i* holds the concentration of stain *i*.

#### `rgb_color_deconvolution(im_rgb, stain_matrix, bg_int=None)`

Convenience wrapper that converts RGB → SDA → concentrations in a single call.

### Reconstruction

#### `reconstruct_rgb(concentrations, stain_matrix, bg_int=None)`

Reconstructs an RGB image from stain concentrations and a stain matrix. Inverts the deconvolution: `SDA = concentrations × Wᵀ`, then converts SDA back to RGB. Useful for stain normalisation workflows where you modify concentration channels and then reconstruct.

See the full [API documentation](https://lavlabinfrastructure.github.io/macenko-pca/) for parameter details.

## Development Setup

**Prerequisites:** Python 3.9+, a [Rust toolchain](https://rustup.rs/), and [Hatch](https://hatch.pypa.io/latest/install/).

On Linux you also need OpenBLAS development headers (`libopenblas-dev` on Debian/Ubuntu). On macOS: `brew install openblas pkg-config`.

```console
git clone https://github.com/LavLabInfrastructure/macenko-pca.git
cd macenko-pca
pip install maturin hatch
maturin develop --release
```

Optionally install pre-commit hooks:

```console
pip install pre-commit
pre-commit install
```

### Common Commands

Run these directly or use the provided [`Makefile`](./Makefile) shortcuts (e.g. `make test`, `make lint`).

| Task | Command |
|------|---------|
| Build Rust extension in-place | `maturin develop --release` |
| Run tests | `make test` |
| Tests + coverage | `make cov` |
| Lint Python | `hatch run lint:check` |
| Format Python | `hatch run lint:format` |
| Auto-fix lint | `hatch run lint:fix` |
| Format + fix + lint | `hatch run lint:all` |
| Type check | `hatch run types:check` |
| Build docs | `hatch run docs:build-docs` |
| Serve docs | `hatch run docs:serve-docs` |
| Build wheel | `maturin build --release` |
| Clean artifacts | `make clean` |
| Rust lints | `make cargo-clippy` |
| Rust tests | `make cargo-test` |

### Docker

```console
# Run tests via Docker
docker build --target maturin -t macenko-pca:maturin .
docker run --rm -e HATCH_ENV=test macenko-pca:maturin cov

# Production image (just the installed wheel)
docker build --target prod -t macenko-pca:prod .
```

## Publishing to PyPI

This project uses [trusted publishing](https://docs.pypi.org/trusted-publishers/) (OIDC) — no API tokens or secrets are needed. The `publish.yml` workflow handles everything automatically.

### One-Time Setup (PyPI)

1. Go to <https://pypi.org/manage/account/publishing/> (logged in as a maintainer of the **lavlab** org).
2. Click **"Add a new pending publisher"** and fill in:
   - **PyPI project name:** `macenko-pca`
   - **Owner:** `lavlab` *(the GitHub organisation)*
   - **Repository:** `macenko-pca`
   - **Workflow name:** `publish.yml`
   - **Environment name:** `pypi`
3. *(Optional)* Repeat on [TestPyPI](https://test.pypi.org/manage/account/publishing/) with environment name `testpypi` to enable dry-run publishes.

### One-Time Setup (GitHub)

1. In the repository settings, go to **Environments**.
2. Create an environment called **`pypi`**.
   - Optionally add a protection rule requiring manual approval before publishing.
3. *(Optional)* Create an environment called **`testpypi`** for test publishes.

### How to Release

```bash
# 1. Bump the version in src/macenko_pca/__about__.py and Cargo.toml
# 2. Commit and tag
git add -A
git commit -m "release: v0.2.0"
git tag v0.2.0
git push && git push --tags

# 3. Create a GitHub Release from the tag (via the web UI or `gh` CLI)
gh release create v0.2.0 --generate-notes
```

Creating the release triggers `publish.yml`, which:
1. Builds wheels on Linux (manylinux), macOS (x86_64 + arm64), and Windows.
2. Builds a source distribution.
3. Publishes everything to PyPI via trusted publishing.

### Testing a Publish (Without a Release)

You can manually trigger the workflow against TestPyPI:

1. Go to **Actions → Publish to PyPI → Run workflow**.
2. Select **`testpypi`** as the target.
3. Verify the package at <https://test.pypi.org/project/macenko-pca/>.

## Project Structure

```text
macenko-pca/
├── src/
│   └── macenko_pca/            # Python package source
│       ├── __init__.py          # Public API & version export
│       ├── __about__.py         # Version string
│       ├── deconvolution.py     # Pythonic wrappers with dtype dispatch
│       └── py.typed             # PEP 561 marker
├── rust/                        # Rust source (compiled via PyO3 + maturin)
│   ├── lib.rs                   # PyO3 module entry point (f32 + f64 variants)
│   ├── float_trait.rs           # MacenkoFloat supertrait (f32/f64)
│   ├── color_conversion.rs      # RGB → SDA transform (generic)
│   ├── color_deconvolution.rs   # SDA → concentrations, RGB reconstruction
│   ├── complement_stain_matrix.rs
│   ├── linalg.rs                # SVD, magnitude, normalisation (generic)
│   ├── rgb_separate_stains_macenko_pca.rs
│   ├── separate_stains_macenko_pca.rs
│   └── utils.rs                 # Image ↔ matrix helpers
├── tests/
│   ├── conftest.py              # Shared pytest fixtures
│   └── test_deconvolution.py    # Library function tests (104 tests)
├── docs/                        # MkDocs source files
├── .github/
│   ├── workflows/
│   │   ├── build.yml            # CI: build wheels on every push/PR
│   │   ├── pytest.yml           # CI: run tests across Python versions
│   │   ├── lint.yml             # CI: ruff lint + format check
│   │   └── publish.yml          # CD: publish to PyPI on release
│   └── dependabot.yml           # Auto-update deps, actions, Docker, Cargo
├── Cargo.toml                   # Rust crate configuration
├── pyproject.toml               # Python project & tool config (maturin backend)
├── Dockerfile                   # Multi-stage build (maturin / dev / prod)
├── Makefile                     # Dev shortcuts
├── mkdocs.yml                   # Docs config
├── .pre-commit-config.yaml      # Pre-commit hooks
├── .editorconfig                # Editor consistency
└── .gitignore
```

## Design Philosophy

This project follows a **library-first** approach:

1. **All logic** lives in importable modules under `src/macenko_pca/`.
2. **Heavy computation** is delegated to Rust (`rust/`) for maximum throughput — SVD via `ndarray-linalg`, parallelism via `rayon`. All Rust functions are generic over `f32`/`f64` via the `MacenkoFloat` trait.
3. **The Python layer** (`deconvolution.py`) detects the input array's dtype and dispatches to the appropriate typed Rust function, providing input validation and rich docstrings.
4. **Tests** call library functions directly.

This keeps your code reusable whether it's called from another package, a Jupyter notebook, or a web API.

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development guidelines.

## License

`macenko-pca` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.