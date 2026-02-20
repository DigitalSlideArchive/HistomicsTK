# Macenko PCA

Welcome to the **macenko-pca** documentation.

## Overview

A high-performance Python package for stain matrix estimation and colour deconvolution of histology images using the Macenko PCA method, with the compute-intensive core written in **Rust** via [PyO3](https://pyo3.rs/).

This package is designed for colour deconvolution of histology images — a common preprocessing step in computational pathology pipelines. It implements the method described in:

> Macenko, M. et al. "A method for normalizing histology slides for quantitative analysis." *ISBI 2009*.

### Features

- **Fast:** Core computation (SVD, PCA projection, angle binning, matrix inversion) runs in compiled Rust with Rayon parallelism.
- **Simple API:** Six functions cover the full workflow — estimate stain vectors, decompose into concentrations, and reconstruct.
- **NumPy native:** Accepts and returns standard NumPy arrays. No special data structures required.
- **Precision-aware:** Pass `float32` arrays to halve RAM usage — the dtype of your input controls which Rust code path runs.

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

If you prefer more control over each step:

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
stain_matrix = rgb_separate_stains_macenko_pca(im_rgb_f32)          # f32
concentrations = rgb_color_deconvolution(im_rgb_f32, stain_matrix)  # f32
reconstructed = reconstruct_rgb(concentrations, stain_matrix)       # f32
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

## API Summary

### Stain Matrix Estimation

| Function | Description |
|----------|-------------|
| `rgb_separate_stains_macenko_pca(im_rgb, ...)` | End-to-end: RGB image → 3×3 stain matrix |
| `separate_stains_macenko_pca(im_sda, ...)` | Lower-level: SDA image → 3×3 stain matrix |

### Colour Conversion

| Function | Description |
|----------|-------------|
| `rgb_to_sda(im_rgb, ...)` | Convert RGB image/matrix to SDA space |

### Colour Deconvolution (Applying Stain Vectors)

| Function | Description |
|----------|-------------|
| `color_deconvolution(im_sda, stain_matrix)` | SDA image + stain matrix → per-stain concentration channels |
| `rgb_color_deconvolution(im_rgb, stain_matrix, ...)` | Convenience: RGB → SDA → concentrations in one call |

### Reconstruction

| Function | Description |
|----------|-------------|
| `reconstruct_rgb(concentrations, stain_matrix, ...)` | Stain concentrations + stain matrix → reconstructed RGB image |

### Development Setup

Clone the repository and install the Rust toolchain and [Hatch](https://hatch.pypa.io/latest/install/):

```console
git clone https://github.com/LavLabInfrastructure/macenko-pca.git
cd macenko-pca
pip install maturin hatch
maturin develop --release
```

### Common Commands

| Task | Command |
|------|---------|
| Build Rust extension in-place | `maturin develop --release` |
| Run tests | `make test` |
| Tests + coverage | `make cov` |
| Lint Python | `hatch run lint:check` |
| Format Python | `hatch run lint:format` |
| Type check | `hatch run types:check` |
| Build wheel | `maturin build --release` |
| Build docs | `hatch run docs:build-docs` |
| Serve docs | `hatch run docs:serve-docs` |
| Rust lints | `make cargo-clippy` |

## Project Structure

```text
macenko-pca/
├── src/
│   └── macenko_pca/            # Python package source
│       ├── __init__.py          # Public API exports
│       ├── __about__.py         # Version info
│       ├── deconvolution.py     # Pythonic wrappers around Rust functions
│       └── py.typed             # PEP 561 type marker
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
├── tests/                       # Test suite (104 tests)
│   ├── conftest.py              # Shared fixtures
│   └── test_deconvolution.py    # Library function tests
├── docs/                        # Documentation source
├── .github/                     # GitHub Actions & Dependabot
│   ├── workflows/
│   │   ├── build.yml            # CI: build wheels on push/PR
│   │   ├── pytest.yml           # CI: tests across Python versions
│   │   ├── lint.yml             # CI: ruff lint + format
│   │   └── publish.yml          # CD: publish to PyPI on release
│   └── dependabot.yml
├── Cargo.toml                   # Rust crate configuration
├── pyproject.toml               # Python project & tool config (maturin backend)
├── Dockerfile                   # Multi-stage Docker build
├── Makefile                     # Common task shortcuts
├── mkdocs.yml                   # Documentation config
└── .pre-commit-config.yaml      # Pre-commit hooks
```

## Design Philosophy

This project follows a **library-first** approach:

1. All business logic lives in importable modules under `src/macenko_pca/`.
2. Heavy computation is delegated to Rust (`rust/`) for maximum performance. All Rust functions are generic over `f32`/`f64` via the `MacenkoFloat` trait.
3. The Python layer (`deconvolution.py`) detects the input array's dtype and dispatches to the appropriate typed Rust function, providing input validation and Pythonic docstrings.
4. Tests call library functions directly.

This makes the code reusable whether imported from another Python package, a Jupyter notebook, or a web API.

## Full API Reference

::: macenko_pca