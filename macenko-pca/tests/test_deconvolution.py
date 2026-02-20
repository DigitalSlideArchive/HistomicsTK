# SPDX-FileCopyrightText: 2024-present barrettMCW <mjbarrett@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Tests for macenko_pca.deconvolution library functions.

Each test class covers a single public function. The ``TestDtypeDispatch``
class at the bottom verifies that float32, float16, and integer inputs are
handled correctly across all three functions.
"""

import numpy as np
import pytest

from macenko_pca.deconvolution import (
    color_deconvolution,
    reconstruct_rgb,
    rgb_color_deconvolution,
    rgb_separate_stains_macenko_pca,
    rgb_to_sda,
    separate_stains_macenko_pca,
)

# ---------------------------------------------------------------------------
# rgb_to_sda
# ---------------------------------------------------------------------------


class TestRgbToSda:
    """Tests for the rgb_to_sda conversion function."""

    def test_basic_shape_3d(self, sample_rgb_image):
        """Output should have the same shape as the 3D input."""
        result = rgb_to_sda(sample_rgb_image)
        assert result.shape == sample_rgb_image.shape

    def test_basic_shape_2d(self, rng):
        """Output should have the same shape as a 2D (N, C) input."""
        im_2d = rng.uniform(1.0, 255.0, size=(100, 3))
        result = rgb_to_sda(im_2d)
        assert result.shape == im_2d.shape

    def test_dtype_preserved_f64(self, sample_rgb_image):
        """float64 input should produce float64 output."""
        result = rgb_to_sda(sample_rgb_image)
        assert result.dtype == np.float64

    def test_dtype_preserved_f32(self, sample_rgb_image_f32):
        """float32 input should produce float32 output."""
        result = rgb_to_sda(sample_rgb_image_f32)
        assert result.dtype == np.float32

    def test_default_non_negative(self, sample_rgb_image):
        """With allow_negative=False (default), all values should be >= 0."""
        result = rgb_to_sda(sample_rgb_image)
        assert np.all(result >= 0.0)

    def test_default_non_negative_f32(self, sample_rgb_image_f32):
        """With allow_negative=False on f32 input, all values should be >= 0."""
        result = rgb_to_sda(sample_rgb_image_f32)
        assert np.all(result >= 0.0)

    def test_allow_negative(self):
        """With allow_negative=True, values below zero should be preserved."""
        im_rgb = np.full((16, 16, 3), 300.0, dtype=np.float64)
        result = rgb_to_sda(im_rgb, bg_int=[256.0], allow_negative=True)
        assert np.any(result < 0.0)

    def test_allow_negative_f32(self):
        """With allow_negative=True on f32 input, negatives should be preserved."""
        im_rgb = np.full((16, 16, 3), 300.0, dtype=np.float32)
        result = rgb_to_sda(im_rgb, bg_int=[256.0], allow_negative=True)
        assert result.dtype == np.float32
        assert np.any(result < 0.0)

    def test_custom_bg_int_single(self, sample_rgb_image):
        """A single-element bg_int should broadcast to all channels."""
        result = rgb_to_sda(sample_rgb_image, bg_int=[240.0])
        assert result.shape == sample_rgb_image.shape

    def test_custom_bg_int_per_channel(self, sample_rgb_image):
        """Per-channel bg_int should be accepted without error."""
        result = rgb_to_sda(sample_rgb_image, bg_int=[240.0, 250.0, 260.0])
        assert result.shape == sample_rgb_image.shape

    def test_custom_bg_int_per_channel_f32(self, sample_rgb_image_f32):
        """Per-channel bg_int should work with f32 input."""
        result = rgb_to_sda(sample_rgb_image_f32, bg_int=[240.0, 250.0, 260.0])
        assert result.shape == sample_rgb_image_f32.shape
        assert result.dtype == np.float32

    def test_invalid_ndim_raises(self):
        """Inputs that are not 2D or 3D should raise ValueError."""
        with pytest.raises(ValueError, match="2D or 3D"):
            rgb_to_sda(np.zeros((5,)))

    def test_integer_input_coerced_to_f64(self, rng):
        """Integer input arrays should be coerced to float64."""
        im_uint8 = rng.integers(1, 255, size=(32, 32, 3), dtype=np.uint8)
        result = rgb_to_sda(im_uint8)
        assert result.dtype == np.float64
        assert result.shape == im_uint8.shape

    def test_f32_and_f64_agree(self, sample_rgb_image, sample_rgb_image_f32):
        """f32 and f64 paths should produce close results."""
        result_64 = rgb_to_sda(sample_rgb_image)
        result_32 = rgb_to_sda(sample_rgb_image_f32)
        np.testing.assert_allclose(
            result_32, result_64.astype(np.float32), atol=1e-2, rtol=1e-3
        )


# ---------------------------------------------------------------------------
# rgb_separate_stains_macenko_pca
# ---------------------------------------------------------------------------


class TestRgbSeparateStainsMacenkoPca:
    """Tests for the high-level RGB -> stain-matrix function."""

    # --- f64 tests ---

    def test_output_shape(self, sample_rgb_image):
        """Result should be a 3x3 stain matrix."""
        result = rgb_separate_stains_macenko_pca(sample_rgb_image)
        assert result.shape == (3, 3)

    def test_output_dtype_f64(self, sample_rgb_image):
        """float64 input should produce float64 output."""
        result = rgb_separate_stains_macenko_pca(sample_rgb_image)
        assert result.dtype == np.float64

    def test_columns_are_normalised(self, sample_rgb_image):
        """Each column (stain vector) should have unit L2 norm."""
        result = rgb_separate_stains_macenko_pca(sample_rgb_image)
        for col_idx in range(3):
            norm = np.linalg.norm(result[:, col_idx])
            assert norm == pytest.approx(1.0, abs=1e-6)

    def test_third_column_is_complement(self, sample_rgb_image):
        """Third stain vector should be the normalised cross product of the first two."""
        result = rgb_separate_stains_macenko_pca(sample_rgb_image)
        cross = np.cross(result[:, 0], result[:, 1])
        cross_normed = cross / np.linalg.norm(cross)
        assert np.allclose(result[:, 2], cross_normed, atol=1e-6) or np.allclose(
            result[:, 2], -cross_normed, atol=1e-6
        )

    def test_with_mask(self, sample_rgb_image, sample_mask):
        """Passing a mask should still produce a valid 3x3 matrix."""
        result = rgb_separate_stains_macenko_pca(sample_rgb_image, mask_out=sample_mask)
        assert result.shape == (3, 3)
        assert np.all(np.isfinite(result))

    def test_custom_bg_int(self, sample_rgb_image):
        """Custom background intensity should produce a valid result."""
        result = rgb_separate_stains_macenko_pca(sample_rgb_image, bg_int=[240.0])
        assert result.shape == (3, 3)
        assert np.all(np.isfinite(result))

    def test_custom_percentiles(self, sample_rgb_image):
        """Adjusting angle percentiles should still produce a valid matrix."""
        result = rgb_separate_stains_macenko_pca(
            sample_rgb_image,
            min_angle_percentile=0.05,
            max_angle_percentile=0.95,
        )
        assert result.shape == (3, 3)
        assert np.all(np.isfinite(result))

    def test_invalid_image_shape_raises(self):
        """A non-3-channel image should raise ValueError."""
        with pytest.raises(ValueError, match="H, W, 3"):
            rgb_separate_stains_macenko_pca(np.zeros((10, 10, 4)))

    def test_invalid_image_ndim_raises(self):
        """A 2D array (not an image) should raise ValueError."""
        with pytest.raises(ValueError, match="H, W, 3"):
            rgb_separate_stains_macenko_pca(np.zeros((100, 3)))

    def test_mismatched_mask_shape_raises(self, sample_rgb_image):
        """A mask whose spatial dims don't match the image should raise."""
        bad_mask = np.zeros((10, 10), dtype=bool)
        with pytest.raises(ValueError, match="mask_out shape"):
            rgb_separate_stains_macenko_pca(sample_rgb_image, mask_out=bad_mask)

    def test_deterministic(self, sample_rgb_image):
        """Repeated calls with the same input should produce the same result."""
        r1 = rgb_separate_stains_macenko_pca(sample_rgb_image)
        r2 = rgb_separate_stains_macenko_pca(sample_rgb_image)
        np.testing.assert_array_equal(r1, r2)

    def test_larger_image(self, sample_rgb_image_large):
        """Smoke-test with a larger image to catch memory/size issues."""
        result = rgb_separate_stains_macenko_pca(sample_rgb_image_large)
        assert result.shape == (3, 3)
        assert np.all(np.isfinite(result))

    # --- f32 tests ---

    def test_output_dtype_f32(self, sample_rgb_image_f32):
        """float32 input should produce float32 output."""
        result = rgb_separate_stains_macenko_pca(sample_rgb_image_f32)
        assert result.dtype == np.float32

    def test_output_shape_f32(self, sample_rgb_image_f32):
        """f32 result should be a 3x3 stain matrix."""
        result = rgb_separate_stains_macenko_pca(sample_rgb_image_f32)
        assert result.shape == (3, 3)

    def test_columns_are_normalised_f32(self, sample_rgb_image_f32):
        """Each column should have unit L2 norm (f32 path)."""
        result = rgb_separate_stains_macenko_pca(sample_rgb_image_f32)
        for col_idx in range(3):
            norm = np.linalg.norm(result[:, col_idx].astype(np.float64))
            assert norm == pytest.approx(1.0, abs=1e-4)

    def test_third_column_is_complement_f32(self, sample_rgb_image_f32):
        """Third stain vector should be the normalised cross product (f32 path)."""
        result = rgb_separate_stains_macenko_pca(sample_rgb_image_f32).astype(
            np.float64
        )
        cross = np.cross(result[:, 0], result[:, 1])
        cross_normed = cross / np.linalg.norm(cross)
        assert np.allclose(result[:, 2], cross_normed, atol=1e-4) or np.allclose(
            result[:, 2], -cross_normed, atol=1e-4
        )

    def test_with_mask_f32(self, sample_rgb_image_f32, sample_mask):
        """Passing a mask should still work with f32 input."""
        result = rgb_separate_stains_macenko_pca(
            sample_rgb_image_f32, mask_out=sample_mask
        )
        assert result.shape == (3, 3)
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))

    def test_deterministic_f32(self, sample_rgb_image_f32):
        """f32 path should be deterministic."""
        r1 = rgb_separate_stains_macenko_pca(sample_rgb_image_f32)
        r2 = rgb_separate_stains_macenko_pca(sample_rgb_image_f32)
        np.testing.assert_array_equal(r1, r2)

    def test_larger_image_f32(self, sample_rgb_image_large_f32):
        """Smoke-test with a larger f32 image."""
        result = rgb_separate_stains_macenko_pca(sample_rgb_image_large_f32)
        assert result.shape == (3, 3)
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))

    def test_f32_and_f64_results_similar(self, sample_rgb_image, sample_rgb_image_f32):
        """f32 and f64 stain matrices should agree within f32 tolerance."""
        r64 = rgb_separate_stains_macenko_pca(sample_rgb_image)
        r32 = rgb_separate_stains_macenko_pca(sample_rgb_image_f32)
        # The stain vectors might differ in sign — compare column-wise
        for col in range(3):
            v64 = r64[:, col]
            v32 = r32[:, col].astype(np.float64)
            close_same = np.allclose(v64, v32, atol=0.05)
            close_flip = np.allclose(v64, -v32, atol=0.05)
            assert close_same or close_flip, (
                f"Column {col} differs too much between f32 and f64:\n"
                f"  f64: {v64}\n  f32: {v32}"
            )


# ---------------------------------------------------------------------------
# separate_stains_macenko_pca (SDA input)
# ---------------------------------------------------------------------------


class TestSeparateStainsMacenkoPca:
    """Tests for the lower-level SDA -> stain-matrix function."""

    # --- f64 ---

    def test_output_shape(self, sample_sda_image):
        """Result should be a 3x3 stain matrix."""
        result = separate_stains_macenko_pca(sample_sda_image)
        assert result.shape == (3, 3)

    def test_output_dtype_f64(self, sample_sda_image):
        """float64 SDA input should produce float64 output."""
        result = separate_stains_macenko_pca(sample_sda_image)
        assert result.dtype == np.float64

    def test_columns_are_normalised(self, sample_sda_image):
        """Each column (stain vector) should have unit L2 norm."""
        result = separate_stains_macenko_pca(sample_sda_image)
        for col_idx in range(3):
            norm = np.linalg.norm(result[:, col_idx])
            assert norm == pytest.approx(1.0, abs=1e-6)

    def test_with_mask(self, sample_sda_image, sample_mask):
        """Passing a mask should still produce a valid 3x3 matrix."""
        result = separate_stains_macenko_pca(sample_sda_image, mask_out=sample_mask)
        assert result.shape == (3, 3)
        assert np.all(np.isfinite(result))

    def test_invalid_ndim_raises(self):
        """A 2D input should raise ValueError."""
        with pytest.raises(ValueError, match="3D"):
            separate_stains_macenko_pca(np.zeros((100, 3)))

    def test_consistency_with_rgb_wrapper(self, sample_rgb_image):
        """Manually converting to SDA then calling the low-level function
        should give the same result as the high-level RGB wrapper."""
        sda = rgb_to_sda(sample_rgb_image)
        result_low = separate_stains_macenko_pca(sda)
        result_high = rgb_separate_stains_macenko_pca(sample_rgb_image)
        np.testing.assert_allclose(result_low, result_high, atol=1e-10)

    # --- f32 ---

    def test_output_dtype_f32(self, sample_sda_image_f32):
        """float32 SDA input should produce float32 output."""
        result = separate_stains_macenko_pca(sample_sda_image_f32)
        assert result.dtype == np.float32

    def test_output_shape_f32(self, sample_sda_image_f32):
        """f32 SDA input should produce a 3x3 matrix."""
        result = separate_stains_macenko_pca(sample_sda_image_f32)
        assert result.shape == (3, 3)

    def test_columns_are_normalised_f32(self, sample_sda_image_f32):
        """Each column should have unit L2 norm (f32 SDA path)."""
        result = separate_stains_macenko_pca(sample_sda_image_f32)
        for col_idx in range(3):
            norm = np.linalg.norm(result[:, col_idx].astype(np.float64))
            assert norm == pytest.approx(1.0, abs=1e-4)

    def test_consistency_with_rgb_wrapper_f32(self, sample_rgb_image_f32):
        """f32 SDA path should match the f32 RGB wrapper path."""
        sda = rgb_to_sda(sample_rgb_image_f32)
        result_low = separate_stains_macenko_pca(sda)
        result_high = rgb_separate_stains_macenko_pca(sample_rgb_image_f32)
        np.testing.assert_allclose(result_low, result_high, atol=1e-5)


# ---------------------------------------------------------------------------
# Dtype dispatch logic
# ---------------------------------------------------------------------------


class TestDtypeDispatch:
    """Tests that dtype detection and promotion work correctly across
    all three public functions."""

    # --- float16 -> promoted to float32 ---

    def test_rgb_to_sda_f16_returns_f32(self, sample_rgb_image_f16):
        """float16 input to rgb_to_sda should produce float32 output."""
        result = rgb_to_sda(sample_rgb_image_f16)
        assert result.dtype == np.float32
        assert result.shape == sample_rgb_image_f16.shape

    def test_rgb_separate_f16_returns_f32(self, sample_rgb_image_f16):
        """float16 input to rgb_separate_stains should produce float32 output."""
        result = rgb_separate_stains_macenko_pca(sample_rgb_image_f16)
        assert result.dtype == np.float32
        assert result.shape == (3, 3)
        assert np.all(np.isfinite(result))

    def test_separate_f16_returns_f32(self, sample_rgb_image_f16):
        """float16 SDA input to separate_stains should produce float32 output."""
        sda = rgb_to_sda(sample_rgb_image_f16)
        assert sda.dtype == np.float32
        result = separate_stains_macenko_pca(sda)
        assert result.dtype == np.float32
        assert result.shape == (3, 3)

    # --- integer -> promoted to float64 ---

    def test_rgb_to_sda_uint8_returns_f64(self, rng):
        """uint8 input should be promoted to float64."""
        im = rng.integers(1, 255, size=(32, 32, 3), dtype=np.uint8)
        result = rgb_to_sda(im)
        assert result.dtype == np.float64
        assert result.shape == im.shape

    def test_rgb_to_sda_int32_returns_f64(self, rng):
        """int32 input should be promoted to float64."""
        im = rng.integers(1, 255, size=(32, 32, 3)).astype(np.int32)
        result = rgb_to_sda(im)
        assert result.dtype == np.float64

    def test_rgb_separate_uint8_returns_f64(self, rng):
        """uint8 input to rgb_separate_stains should produce float64 output."""
        im = rng.integers(1, 255, size=(64, 64, 3), dtype=np.uint8)
        result = rgb_separate_stains_macenko_pca(im)
        assert result.dtype == np.float64
        assert result.shape == (3, 3)

    def test_rgb_separate_int16_returns_f64(self, rng):
        """int16 input should be promoted to float64."""
        im = rng.integers(1, 255, size=(64, 64, 3)).astype(np.int16)
        result = rgb_separate_stains_macenko_pca(im)
        assert result.dtype == np.float64
        assert result.shape == (3, 3)

    # --- explicit float32 preserved ---

    def test_rgb_to_sda_f32_stays_f32(self, sample_rgb_image_f32):
        """Explicit float32 input should NOT be promoted to float64."""
        result = rgb_to_sda(sample_rgb_image_f32)
        assert result.dtype == np.float32

    def test_rgb_separate_f32_stays_f32(self, sample_rgb_image_f32):
        """Explicit float32 input should NOT be promoted to float64."""
        result = rgb_separate_stains_macenko_pca(sample_rgb_image_f32)
        assert result.dtype == np.float32

    # --- explicit float64 preserved ---

    def test_rgb_to_sda_f64_stays_f64(self, sample_rgb_image):
        """Explicit float64 input stays float64 (no downcast)."""
        result = rgb_to_sda(sample_rgb_image)
        assert result.dtype == np.float64

    def test_rgb_separate_f64_stays_f64(self, sample_rgb_image):
        """Explicit float64 input stays float64 (no downcast)."""
        result = rgb_separate_stains_macenko_pca(sample_rgb_image)
        assert result.dtype == np.float64

    # --- 2D matrix inputs ---

    def test_rgb_to_sda_2d_f32(self, rng):
        """2D float32 matrix should be processed in f32."""
        im_2d = rng.uniform(1.0, 255.0, size=(100, 3)).astype(np.float32)
        result = rgb_to_sda(im_2d)
        assert result.dtype == np.float32
        assert result.shape == im_2d.shape

    def test_rgb_to_sda_2d_f64(self, rng):
        """2D float64 matrix should be processed in f64."""
        im_2d = rng.uniform(1.0, 255.0, size=(100, 3))
        result = rgb_to_sda(im_2d)
        assert result.dtype == np.float64
        assert result.shape == im_2d.shape


# ---------------------------------------------------------------------------
# Integration / end-to-end
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# color_deconvolution
# ---------------------------------------------------------------------------


class TestColorDeconvolution:
    """Tests for color_deconvolution (SDA → concentrations)."""

    def test_output_shape_f64(self, sample_rgb_image):
        """Output shape should match the input spatial dims (f64)."""
        sda = rgb_to_sda(sample_rgb_image)
        w = rgb_separate_stains_macenko_pca(sample_rgb_image)
        result = color_deconvolution(sda, w)
        assert result.shape == sample_rgb_image.shape

    def test_output_dtype_f64(self, sample_rgb_image):
        """Output dtype should be float64 when input is float64."""
        sda = rgb_to_sda(sample_rgb_image)
        w = rgb_separate_stains_macenko_pca(sample_rgb_image)
        result = color_deconvolution(sda, w)
        assert result.dtype == np.float64

    def test_output_shape_f32(self, sample_rgb_image_f32):
        """Output shape should match the input spatial dims (f32)."""
        sda = rgb_to_sda(sample_rgb_image_f32)
        w = rgb_separate_stains_macenko_pca(sample_rgb_image_f32)
        result = color_deconvolution(sda, w)
        assert result.shape == sample_rgb_image_f32.shape

    def test_output_dtype_f32(self, sample_rgb_image_f32):
        """Output dtype should be float32 when input is float32."""
        sda = rgb_to_sda(sample_rgb_image_f32)
        w = rgb_separate_stains_macenko_pca(sample_rgb_image_f32)
        result = color_deconvolution(sda, w)
        assert result.dtype == np.float32

    def test_all_finite(self, sample_rgb_image):
        """All concentration values should be finite."""
        sda = rgb_to_sda(sample_rgb_image)
        w = rgb_separate_stains_macenko_pca(sample_rgb_image)
        result = color_deconvolution(sda, w)
        assert np.all(np.isfinite(result))

    def test_identity_stain_matrix(self, sample_rgb_image):
        """With an identity stain matrix, concentrations should equal the SDA image."""
        sda = rgb_to_sda(sample_rgb_image)
        identity = np.eye(3, dtype=np.float64)
        result = color_deconvolution(sda, identity)
        np.testing.assert_allclose(result, sda, atol=1e-10)

    def test_invalid_image_shape_raises(self, rng):
        """Should raise ValueError for non-(H,W,3) input."""
        bad = rng.uniform(0, 255, size=(64, 64, 4))
        w = np.eye(3)
        with pytest.raises(ValueError, match="H, W, 3"):
            color_deconvolution(bad, w)

    def test_invalid_image_ndim_raises(self, rng):
        """Should raise ValueError for 2D input."""
        bad = rng.uniform(0, 255, size=(64, 3))
        w = np.eye(3)
        with pytest.raises(ValueError, match="H, W, 3"):
            color_deconvolution(bad, w)

    def test_invalid_stain_matrix_shape_raises(self, sample_rgb_image):
        """Should raise ValueError for non-3x3 stain matrix."""
        sda = rgb_to_sda(sample_rgb_image)
        bad_w = np.eye(4)
        with pytest.raises(ValueError, match="3, 3"):
            color_deconvolution(sda, bad_w)

    def test_deterministic(self, sample_rgb_image):
        """Repeated calls with the same input should return identical results."""
        sda = rgb_to_sda(sample_rgb_image)
        w = rgb_separate_stains_macenko_pca(sample_rgb_image)
        r1 = color_deconvolution(sda, w)
        r2 = color_deconvolution(sda, w)
        np.testing.assert_array_equal(r1, r2)

    def test_f16_input_promoted(self, sample_rgb_image_f16):
        """float16 input should be promoted to float32."""
        sda = rgb_to_sda(sample_rgb_image_f16)
        w = rgb_separate_stains_macenko_pca(sample_rgb_image_f16)
        result = color_deconvolution(sda, w)
        assert result.dtype == np.float32

    def test_integer_input_promoted(self, rng):
        """Integer input should be promoted to float64."""
        im_uint8 = rng.integers(1, 255, size=(64, 64, 3), dtype=np.uint8)
        sda = rgb_to_sda(im_uint8)
        w = rgb_separate_stains_macenko_pca(im_uint8)
        result = color_deconvolution(sda, w)
        assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# rgb_color_deconvolution
# ---------------------------------------------------------------------------


class TestRgbColorDeconvolution:
    """Tests for rgb_color_deconvolution (RGB → concentrations)."""

    def test_output_shape_f64(self, sample_rgb_image):
        """Output shape should match the input spatial dims (f64)."""
        w = rgb_separate_stains_macenko_pca(sample_rgb_image)
        result = rgb_color_deconvolution(sample_rgb_image, w)
        assert result.shape == sample_rgb_image.shape

    def test_output_dtype_f64(self, sample_rgb_image):
        """Output dtype should be float64 when input is float64."""
        w = rgb_separate_stains_macenko_pca(sample_rgb_image)
        result = rgb_color_deconvolution(sample_rgb_image, w)
        assert result.dtype == np.float64

    def test_output_shape_f32(self, sample_rgb_image_f32):
        """Output shape should match the input spatial dims (f32)."""
        w = rgb_separate_stains_macenko_pca(sample_rgb_image_f32)
        result = rgb_color_deconvolution(sample_rgb_image_f32, w)
        assert result.shape == sample_rgb_image_f32.shape

    def test_output_dtype_f32(self, sample_rgb_image_f32):
        """Output dtype should be float32 when input is float32."""
        w = rgb_separate_stains_macenko_pca(sample_rgb_image_f32)
        result = rgb_color_deconvolution(sample_rgb_image_f32, w)
        assert result.dtype == np.float32

    def test_all_finite(self, sample_rgb_image):
        """All concentration values should be finite."""
        w = rgb_separate_stains_macenko_pca(sample_rgb_image)
        result = rgb_color_deconvolution(sample_rgb_image, w)
        assert np.all(np.isfinite(result))

    def test_consistency_with_manual_pipeline(self, sample_rgb_image):
        """Result should match manual rgb_to_sda + color_deconvolution."""
        w = rgb_separate_stains_macenko_pca(sample_rgb_image)
        # Manual two-step
        sda = rgb_to_sda(sample_rgb_image)
        manual = color_deconvolution(sda, w)
        # One-step convenience
        auto = rgb_color_deconvolution(sample_rgb_image, w)
        np.testing.assert_allclose(auto, manual, atol=1e-10)

    def test_consistency_with_manual_pipeline_f32(self, sample_rgb_image_f32):
        """Result should match manual rgb_to_sda + color_deconvolution (f32)."""
        w = rgb_separate_stains_macenko_pca(sample_rgb_image_f32)
        sda = rgb_to_sda(sample_rgb_image_f32)
        manual = color_deconvolution(sda, w)
        auto = rgb_color_deconvolution(sample_rgb_image_f32, w)
        np.testing.assert_allclose(auto, manual, atol=1e-4)

    def test_custom_bg_int(self, sample_rgb_image):
        """Custom bg_int should not crash and should produce finite output."""
        w = rgb_separate_stains_macenko_pca(sample_rgb_image)
        result = rgb_color_deconvolution(sample_rgb_image, w, bg_int=[240.0])
        assert result.shape == sample_rgb_image.shape
        assert np.all(np.isfinite(result))

    def test_invalid_image_shape_raises(self, rng):
        """Should raise ValueError for non-(H,W,3) input."""
        bad = rng.uniform(0, 255, size=(64, 64, 4))
        w = np.eye(3)
        with pytest.raises(ValueError, match="H, W, 3"):
            rgb_color_deconvolution(bad, w)

    def test_invalid_stain_matrix_shape_raises(self, sample_rgb_image):
        """Should raise ValueError for non-3x3 stain matrix."""
        bad_w = np.eye(4)
        with pytest.raises(ValueError, match="3, 3"):
            rgb_color_deconvolution(sample_rgb_image, bad_w)

    def test_deterministic(self, sample_rgb_image):
        """Repeated calls should return identical results."""
        w = rgb_separate_stains_macenko_pca(sample_rgb_image)
        r1 = rgb_color_deconvolution(sample_rgb_image, w)
        r2 = rgb_color_deconvolution(sample_rgb_image, w)
        np.testing.assert_array_equal(r1, r2)

    def test_f16_input_promoted(self, sample_rgb_image_f16):
        """float16 input should be promoted to float32."""
        w = rgb_separate_stains_macenko_pca(sample_rgb_image_f16)
        result = rgb_color_deconvolution(sample_rgb_image_f16, w)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# reconstruct_rgb
# ---------------------------------------------------------------------------


class TestReconstructRgb:
    """Tests for reconstruct_rgb (concentrations → RGB)."""

    def test_output_shape_f64(self, sample_rgb_image):
        """Output shape should match the original image (f64)."""
        w = rgb_separate_stains_macenko_pca(sample_rgb_image)
        conc = rgb_color_deconvolution(sample_rgb_image, w)
        result = reconstruct_rgb(conc, w)
        assert result.shape == sample_rgb_image.shape

    def test_output_dtype_f64(self, sample_rgb_image):
        """Output dtype should be float64 when input is float64."""
        w = rgb_separate_stains_macenko_pca(sample_rgb_image)
        conc = rgb_color_deconvolution(sample_rgb_image, w)
        result = reconstruct_rgb(conc, w)
        assert result.dtype == np.float64

    def test_output_shape_f32(self, sample_rgb_image_f32):
        """Output shape should match the original image (f32)."""
        w = rgb_separate_stains_macenko_pca(sample_rgb_image_f32)
        conc = rgb_color_deconvolution(sample_rgb_image_f32, w)
        result = reconstruct_rgb(conc, w)
        assert result.shape == sample_rgb_image_f32.shape

    def test_output_dtype_f32(self, sample_rgb_image_f32):
        """Output dtype should be float32 when input is float32."""
        w = rgb_separate_stains_macenko_pca(sample_rgb_image_f32)
        conc = rgb_color_deconvolution(sample_rgb_image_f32, w)
        result = reconstruct_rgb(conc, w)
        assert result.dtype == np.float32

    def test_all_finite(self, sample_rgb_image):
        """All reconstructed values should be finite."""
        w = rgb_separate_stains_macenko_pca(sample_rgb_image)
        conc = rgb_color_deconvolution(sample_rgb_image, w)
        result = reconstruct_rgb(conc, w)
        assert np.all(np.isfinite(result))

    def test_values_non_negative(self, sample_rgb_image):
        """Reconstructed RGB values should be non-negative."""
        w = rgb_separate_stains_macenko_pca(sample_rgb_image)
        conc = rgb_color_deconvolution(sample_rgb_image, w)
        result = reconstruct_rgb(conc, w)
        assert np.all(result >= 0.0)

    def test_values_clamped_to_bg(self, sample_rgb_image):
        """Reconstructed RGB values should not exceed bg_int (default 256)."""
        w = rgb_separate_stains_macenko_pca(sample_rgb_image)
        conc = rgb_color_deconvolution(sample_rgb_image, w)
        result = reconstruct_rgb(conc, w)
        assert np.all(result <= 256.0)

    def test_roundtrip_f64(self, sample_rgb_image):
        """Deconvolution then reconstruction should approximate the original image."""
        w = rgb_separate_stains_macenko_pca(sample_rgb_image)
        conc = rgb_color_deconvolution(sample_rgb_image, w)
        reconstructed = reconstruct_rgb(conc, w)
        # The SDA transform clamps negative values to zero, so the
        # round-trip is lossy for very small pixel values.  Use a
        # generous tolerance.
        np.testing.assert_allclose(reconstructed, sample_rgb_image, atol=5.0, rtol=0.05)

    def test_roundtrip_f32(self, sample_rgb_image_f32):
        """Deconvolution then reconstruction should approximate the original (f32)."""
        w = rgb_separate_stains_macenko_pca(sample_rgb_image_f32)
        conc = rgb_color_deconvolution(sample_rgb_image_f32, w)
        reconstructed = reconstruct_rgb(conc, w)
        np.testing.assert_allclose(
            reconstructed, sample_rgb_image_f32, atol=5.0, rtol=0.05
        )

    def test_custom_bg_int(self, sample_rgb_image):
        """Custom bg_int should produce finite output clamped to that value."""
        w = rgb_separate_stains_macenko_pca(sample_rgb_image)
        conc = rgb_color_deconvolution(sample_rgb_image, w)
        result = reconstruct_rgb(conc, w, bg_int=240.0)
        assert result.shape == sample_rgb_image.shape
        assert np.all(np.isfinite(result))
        assert np.all(result <= 240.0)

    def test_zero_stain_channel_reconstruction(self, sample_rgb_image):
        """Zeroing a stain channel and reconstructing should still produce valid RGB."""
        w = rgb_separate_stains_macenko_pca(sample_rgb_image)
        conc = rgb_color_deconvolution(sample_rgb_image, w)
        # Zero-out the second stain (e.g. eosin)
        conc_modified = conc.copy()
        conc_modified[:, :, 1] = 0.0
        result = reconstruct_rgb(conc_modified, w)
        assert result.shape == sample_rgb_image.shape
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0)

    def test_invalid_concentrations_shape_raises(self, rng):
        """Should raise ValueError for non-(H,W,3) concentrations."""
        bad = rng.uniform(0, 1, size=(64, 64, 4))
        w = np.eye(3)
        with pytest.raises(ValueError, match="H, W, 3"):
            reconstruct_rgb(bad, w)

    def test_invalid_stain_matrix_shape_raises(self, sample_rgb_image):
        """Should raise ValueError for non-3x3 stain matrix."""
        w = rgb_separate_stains_macenko_pca(sample_rgb_image)
        conc = rgb_color_deconvolution(sample_rgb_image, w)
        bad_w = np.eye(4)
        with pytest.raises(ValueError, match="3, 3"):
            reconstruct_rgb(conc, bad_w)

    def test_deterministic(self, sample_rgb_image):
        """Repeated calls should return identical results."""
        w = rgb_separate_stains_macenko_pca(sample_rgb_image)
        conc = rgb_color_deconvolution(sample_rgb_image, w)
        r1 = reconstruct_rgb(conc, w)
        r2 = reconstruct_rgb(conc, w)
        np.testing.assert_array_equal(r1, r2)

    def test_f16_input_promoted(self, sample_rgb_image_f16):
        """float16 concentrations should be promoted to float32."""
        w = rgb_separate_stains_macenko_pca(sample_rgb_image_f16)
        conc = rgb_color_deconvolution(sample_rgb_image_f16, w)
        conc_f16 = conc.astype(np.float16)
        result = reconstruct_rgb(conc_f16, w)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests combining multiple library functions."""

    def test_full_pipeline_f64(self, sample_rgb_image):
        """Run the full pipeline: RGB -> SDA -> stain matrix (f64)."""
        sda = rgb_to_sda(sample_rgb_image)
        assert sda.shape == sample_rgb_image.shape
        assert sda.dtype == np.float64

        stain_matrix = separate_stains_macenko_pca(sda)
        assert stain_matrix.shape == (3, 3)
        assert stain_matrix.dtype == np.float64
        assert np.all(np.isfinite(stain_matrix))

    def test_full_pipeline_f32(self, sample_rgb_image_f32):
        """Run the full pipeline: RGB -> SDA -> stain matrix (f32)."""
        sda = rgb_to_sda(sample_rgb_image_f32)
        assert sda.shape == sample_rgb_image_f32.shape
        assert sda.dtype == np.float32

        stain_matrix = separate_stains_macenko_pca(sda)
        assert stain_matrix.shape == (3, 3)
        assert stain_matrix.dtype == np.float32
        assert np.all(np.isfinite(stain_matrix))

    def test_full_pipeline_f16(self, sample_rgb_image_f16):
        """Run the full pipeline starting from f16: promoted to f32."""
        sda = rgb_to_sda(sample_rgb_image_f16)
        assert sda.dtype == np.float32

        stain_matrix = separate_stains_macenko_pca(sda)
        assert stain_matrix.shape == (3, 3)
        assert stain_matrix.dtype == np.float32
        assert np.all(np.isfinite(stain_matrix))

    def test_stain_matrix_is_orthogonal_ish_f64(self, sample_rgb_image):
        """The stain matrix should be close to orthogonal (f64)."""
        result = rgb_separate_stains_macenko_pca(sample_rgb_image)
        dot_02 = np.dot(result[:, 0], result[:, 2])
        dot_12 = np.dot(result[:, 1], result[:, 2])
        # Columns 0 & 2 and 1 & 2 should be orthogonal by construction
        assert dot_02 == pytest.approx(0.0, abs=1e-6)
        assert dot_12 == pytest.approx(0.0, abs=1e-6)
        # Columns 0 & 1 are data-driven, so just check they are finite
        assert np.isfinite(np.dot(result[:, 0], result[:, 1]))

    def test_stain_matrix_is_orthogonal_ish_f32(self, sample_rgb_image_f32):
        """The stain matrix should be close to orthogonal (f32)."""
        result = rgb_separate_stains_macenko_pca(sample_rgb_image_f32).astype(
            np.float64
        )
        dot_02 = np.dot(result[:, 0], result[:, 2])
        dot_12 = np.dot(result[:, 1], result[:, 2])
        # Looser tolerance for f32
        assert dot_02 == pytest.approx(0.0, abs=1e-4)
        assert dot_12 == pytest.approx(0.0, abs=1e-4)
        assert np.isfinite(np.dot(result[:, 0], result[:, 1]))

    def test_full_deconvolution_roundtrip_f64(self, sample_rgb_image):
        """Full pipeline: estimate stains → deconvolve → reconstruct (f64)."""
        w = rgb_separate_stains_macenko_pca(sample_rgb_image)
        conc = rgb_color_deconvolution(sample_rgb_image, w)
        recon = reconstruct_rgb(conc, w)
        assert recon.shape == sample_rgb_image.shape
        assert recon.dtype == np.float64
        # Round-trip should be close (lossy due to SDA clamping)
        np.testing.assert_allclose(recon, sample_rgb_image, atol=5.0, rtol=0.05)

    def test_full_deconvolution_roundtrip_f32(self, sample_rgb_image_f32):
        """Full pipeline: estimate stains → deconvolve → reconstruct (f32)."""
        w = rgb_separate_stains_macenko_pca(sample_rgb_image_f32)
        conc = rgb_color_deconvolution(sample_rgb_image_f32, w)
        recon = reconstruct_rgb(conc, w)
        assert recon.shape == sample_rgb_image_f32.shape
        assert recon.dtype == np.float32
        np.testing.assert_allclose(recon, sample_rgb_image_f32, atol=5.0, rtol=0.05)

    def test_stain_isolation_produces_different_images(self, sample_rgb_image):
        """Isolating different stains should produce visually different reconstructions."""
        w = rgb_separate_stains_macenko_pca(sample_rgb_image)
        conc = rgb_color_deconvolution(sample_rgb_image, w)

        # Isolate stain 0 only
        conc_s0 = conc.copy()
        conc_s0[:, :, 1] = 0.0
        conc_s0[:, :, 2] = 0.0
        recon_s0 = reconstruct_rgb(conc_s0, w)

        # Isolate stain 1 only
        conc_s1 = conc.copy()
        conc_s1[:, :, 0] = 0.0
        conc_s1[:, :, 2] = 0.0
        recon_s1 = reconstruct_rgb(conc_s1, w)

        # The two isolated images should differ
        assert not np.allclose(recon_s0, recon_s1, atol=1.0)
