# SPDX-FileCopyrightText: 2024-present barrettMCW <mjbarrett@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Shared pytest fixtures and configuration for macenko_pca tests."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Provide a seeded random number generator for reproducible tests.

    :return: numpy random generator with fixed seed
    :rtype: numpy.random.Generator
    """
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# float64 (default)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_rgb_image(rng):
    """Provide a small synthetic RGB image for testing (float64).

    Returns a 64×64×3 float64 image with values in [1, 255].

    :return: synthetic RGB image
    :rtype: numpy.ndarray
    """
    return rng.uniform(1.0, 255.0, size=(64, 64, 3))


@pytest.fixture
def sample_rgb_image_large(rng):
    """Provide a larger synthetic RGB image for more realistic testing (float64).

    Returns a 256×256×3 float64 image with values in [1, 255].

    :return: synthetic RGB image
    :rtype: numpy.ndarray
    """
    return rng.uniform(1.0, 255.0, size=(256, 256, 3))


# ---------------------------------------------------------------------------
# float32
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_rgb_image_f32(sample_rgb_image):
    """Provide a small synthetic RGB image as float32.

    Derived from the float64 fixture by casting.

    :return: synthetic RGB image (float32)
    :rtype: numpy.ndarray
    """
    return sample_rgb_image.astype(np.float32)


@pytest.fixture
def sample_rgb_image_large_f32(sample_rgb_image_large):
    """Provide a larger synthetic RGB image as float32.

    :return: synthetic RGB image (float32)
    :rtype: numpy.ndarray
    """
    return sample_rgb_image_large.astype(np.float32)


# ---------------------------------------------------------------------------
# float16
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_rgb_image_f16(sample_rgb_image):
    """Provide a small synthetic RGB image as float16.

    float16 inputs are expected to be promoted to float32 internally.

    :return: synthetic RGB image (float16)
    :rtype: numpy.ndarray
    """
    return sample_rgb_image.astype(np.float16)


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_mask():
    """Provide a boolean mask matching the small sample image dimensions.

    Creates a 64×64 mask where a central 16×16 block is True (excluded).

    :return: boolean mask array
    :rtype: numpy.ndarray
    """
    mask = np.zeros((64, 64), dtype=bool)
    mask[24:40, 24:40] = True
    return mask


# ---------------------------------------------------------------------------
# SDA images (derived from RGB via rgb_to_sda)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_sda_image(sample_rgb_image):
    """Provide a small synthetic SDA image derived from the sample RGB image (float64).

    Uses the rgb_to_sda conversion with default parameters.

    :return: synthetic SDA image with shape (64, 64, 3)
    :rtype: numpy.ndarray
    """
    from macenko_pca.deconvolution import rgb_to_sda

    return rgb_to_sda(sample_rgb_image)


@pytest.fixture
def sample_sda_image_f32(sample_rgb_image_f32):
    """Provide a small synthetic SDA image derived from the f32 RGB image.

    :return: synthetic SDA image with shape (64, 64, 3), dtype float32
    :rtype: numpy.ndarray
    """
    from macenko_pca.deconvolution import rgb_to_sda

    return rgb_to_sda(sample_rgb_image_f32)
