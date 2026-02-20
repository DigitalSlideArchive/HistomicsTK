ARG PY_VERSION=3.12

# ---------------------------------------------------------------------------
# Base: Python + Rust toolchain + source
# ---------------------------------------------------------------------------
FROM python:${PY_VERSION}-slim AS base

# System deps required by OpenBLAS / ndarray-linalg and the Rust toolchain
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    pkg-config \
    libopenblas-dev \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Rust via rustup
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH="/usr/local/cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable \
    && rustup show

# Create non-root user (primarily for devcontainer)
RUN groupadd --gid 1000 vscode \
    && useradd --uid 1000 --gid 1000 -m vscode

WORKDIR /app

# Copy only the files needed for a build (layer caching friendly)
COPY Cargo.toml pyproject.toml README.md LICENSE.txt ./
COPY rust/ rust/
COPY src/ src/

RUN chown -R vscode:vscode /app

# ---------------------------------------------------------------------------
# Maturin: build helper image — use as entrypoint for CI tasks
# ---------------------------------------------------------------------------
FROM base AS maturin

RUN pip3 install --no-cache-dir maturin hatch hatch-pip-compile

# Build the wheel so the extension is importable
RUN maturin develop --release

ENV HATCH_ENV=default
ENTRYPOINT ["hatch", "run"]

# ---------------------------------------------------------------------------
# Dev: full development environment
# ---------------------------------------------------------------------------
FROM base AS dev

COPY tests/ tests/
COPY docs/ docs/
COPY mkdocs.yml Makefile ./
COPY requirements/ requirements/

RUN pip3 install --no-cache-dir maturin hatch hatch-pip-compile \
    && maturin develop --release

USER vscode

# ---------------------------------------------------------------------------
# Prod: minimal production image with only the installed wheel
# ---------------------------------------------------------------------------
FROM python:${PY_VERSION}-slim AS prod

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 vscode \
    && useradd --uid 1000 --gid 1000 -m vscode

# Install maturin only to build the wheel, then remove it
COPY --from=maturin /app /tmp/app
RUN pip3 install --no-cache-dir maturin \
    && cd /tmp/app \
    && maturin build --release \
    && pip3 install --no-cache-dir /tmp/app/target/wheels/*.whl \
    && pip3 uninstall -y maturin \
    && rm -rf /tmp/app

USER vscode
WORKDIR /app

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import macenko_pca" || exit 1
