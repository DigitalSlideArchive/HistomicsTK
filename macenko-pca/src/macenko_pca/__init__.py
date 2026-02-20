# SPDX-FileCopyrightText: 2024-present barrettMCW <mjbarrett@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Macenko PCA stain deconvolution — high-performance Rust-backed implementation.

This package provides fast stain matrix estimation and colour deconvolution
for histology images using the Macenko PCA method, with the compute-intensive
core written in Rust via PyO3.

The input array's dtype controls which precision path is taken:

- ``float64`` → Rust f64 pipeline
- ``float32`` → Rust f32 pipeline (≈ half the RAM)
- ``float16`` → promoted to float32, then Rust f32 pipeline (no f16 LAPACK)
- integer types → promoted to float64 for backward compatibility

Example::

    import numpy as np
    from macenko_pca import rgb_separate_stains_macenko_pca, rgb_color_deconvolution

    # Compute stain matrix from an RGB image
    im_rgb = np.random.rand(256, 256, 3) * 255.0
    stain_matrix = rgb_separate_stains_macenko_pca(im_rgb)

    # Decompose into per-stain concentration channels
    concentrations = rgb_color_deconvolution(im_rgb, stain_matrix)

    # Reconstruct with a stain removed
    from macenko_pca import reconstruct_rgb

    concentrations[:, :, 1] = 0.0  # zero-out eosin
    hematoxylin_only = reconstruct_rgb(concentrations, stain_matrix)
"""

from macenko_pca.__about__ import __version__
from macenko_pca.deconvolution import (
    color_deconvolution,
    reconstruct_rgb,
    rgb_color_deconvolution,
    rgb_separate_stains_macenko_pca,
    rgb_to_sda,
    separate_stains_macenko_pca,
)

__all__ = [
    "__version__",
    "color_deconvolution",
    "reconstruct_rgb",
    "rgb_color_deconvolution",
    "rgb_separate_stains_macenko_pca",
    "rgb_to_sda",
    "separate_stains_macenko_pca",
]
