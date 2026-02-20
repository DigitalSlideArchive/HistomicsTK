# Contributing

Thank you for your interest in contributing! This document provides guidelines for working with this project.

## Development Setup

1. **Clone the repository:**

   ```console
   git clone https://github.com/LavLabInfrastructure/macenko-pca.git
   cd macenko-pca
   ```

2. **Install the Rust toolchain** (if you don't have it already):

   ```console
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

3. **Install system dependencies:**

   On Debian/Ubuntu:

   ```console
   sudo apt-get install libopenblas-dev pkg-config
   ```

   On macOS:

   ```console
   brew install openblas pkg-config
   ```

4. **Install Python tooling** ([maturin](https://github.com/PyO3/maturin) and [Hatch](https://hatch.pypa.io/)):

   ```console
   pip install maturin hatch
   ```

5. **Build the Rust extension in-place:**

   ```console
   maturin develop --release
   ```

6. **Install pre-commit hooks:**

   ```console
   pip install pre-commit
   pre-commit install
   ```

## Common Tasks

Python environment tasks are managed through [Hatch environments](https://hatch.pypa.io/latest/environment/). Rust builds go through maturin. The [`Makefile`](./Makefile) wraps both for convenience.

| Task                         | Command                          |
|------------------------------|----------------------------------|
| Build Rust extension in-place | `maturin develop --release`     |
| Run tests                    | `make test`                      |
| Run tests with coverage      | `make cov`                       |
| Lint Python code             | `hatch run lint:check`           |
| Format Python code           | `hatch run lint:format`          |
| Auto-fix lint issues         | `hatch run lint:fix`             |
| Run all Python lint checks   | `hatch run lint:all`             |
| Type check                   | `hatch run types:check`          |
| Build docs                   | `hatch run docs:build-docs`      |
| Serve docs locally           | `hatch run docs:serve-docs`      |
| Build wheel                  | `maturin build --release`        |
| Rust lints (clippy)          | `make cargo-clippy`              |
| Rust unit tests              | `make cargo-test`                |
| Clean all artifacts          | `make clean`                     |

## Code Style

### Python

- **Formatter & Linter:** [Ruff](https://docs.astral.sh/ruff/) handles both formatting and linting. Configuration lives in `pyproject.toml` under `[tool.ruff]`.
- **Line length:** 88 characters (consistent with Black's default).
- **Docstrings:** Use [Sphinx-style](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html) docstrings (`:param:`, `:type:`, `:return:`, `:rtype:`).
- **Type hints:** Encouraged. The project includes a `py.typed` marker for PEP 561 compliance.

### Rust

- **Edition:** Rust 2024 edition.
- **Linter:** `cargo clippy -- -D warnings` — treat all warnings as errors.
- **Formatting:** `cargo fmt` (standard rustfmt defaults).
- **Dependencies:** Keep the dependency tree lean. Prefer well-established crates (ndarray, rayon, PyO3).

## Project Architecture

This project follows a **library-first** design with a Rust performance core:

1. **Rust modules** (`rust/`) contain the compute-intensive logic — SVD, PCA projection, colour conversion, and stain matrix assembly. These are exposed to Python via PyO3 bindings in `rust/lib.rs`.
2. **Python library modules** (`src/macenko_pca/`) provide Pythonic wrappers with input validation, dtype coercion, and full docstrings. The heavy lifting is delegated to the compiled Rust extension (`macenko_pca._rust`).
3. **CLI** (`src/macenko_pca/cli.py`) is a thin wrapper that parses arguments and delegates to library functions.
4. **Tests** (`tests/`) import directly from the Python library — never from the CLI.

When adding new functionality:

- If it is compute-intensive, implement it in Rust and expose a `#[pyfunction]` in `rust/lib.rs`.
- Write a Python wrapper in `deconvolution.py` (or a new module) that validates inputs and calls the Rust function.
- Add tests that call the Python function directly.
- If CLI access is needed, add a subcommand in `cli.py` that wraps the library function.

## Writing Tests

- Tests live in `tests/` and use [pytest](https://docs.pytest.org/).
- Shared fixtures belong in `tests/conftest.py`.
- Name test files `test_<module>.py` and test functions `test_<behavior>`.
- Aim for descriptive test names that explain what is being verified.
- **Important:** You must build the Rust extension before running tests (`maturin develop --release`), or use `make test` which does this automatically.

```python
def test_stain_matrix_is_3x3(sample_rgb_image):
    """Verify the stain matrix has the expected shape."""
    from macenko_pca import rgb_separate_stains_macenko_pca
    result = rgb_separate_stains_macenko_pca(sample_rgb_image)
    assert result.shape == (3, 3)
```

## Submitting Changes

1. **Create a branch** from `main`:

   ```console
   git checkout -b feature/my-change
   ```

2. **Make your changes** and ensure all checks pass:

   ```console
   maturin develop --release
   hatch run lint:all
   make test
   hatch run types:check
   make cargo-clippy
   ```

3. **Commit** with a clear, descriptive message. Pre-commit hooks will run Ruff automatically.

4. **Open a Pull Request** against `main`. The CI pipeline will run linting, tests, and wheel builds across multiple platforms.

## Docker

The project includes multi-stage Docker builds:

| Target    | Purpose                                             |
|-----------|-----------------------------------------------------|
| `base`    | Base image with Rust toolchain, source, and non-root user |
| `maturin` | Runs Hatch commands with the extension pre-built (used in CI) |
| `dev`     | Full development environment                        |
| `prod`    | Minimal production image with only the wheel installed |

```console
# Run tests via Docker
docker build --target maturin -t macenko-pca:maturin .
docker run --rm -e HATCH_ENV=test macenko-pca:maturin cov

# Build production image
docker build --target prod -t macenko-pca:prod .
```

## Questions?

Open an issue on GitHub if you have questions or run into problems.