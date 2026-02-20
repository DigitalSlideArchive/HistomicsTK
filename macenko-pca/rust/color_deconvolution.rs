use crate::color_conversion::rgb_to_sda;
use crate::float_trait::MacenkoFloat;
use ndarray::{Array3, ArrayView2, ArrayView3};
use ndarray_linalg::Inverse;
use num_traits::Float;
use rayon::prelude::*;

/// Apply a stain matrix to deconvolve an SDA image into stain concentrations.
///
/// Given an SDA image and a stain matrix W (3×3, columns = stain vectors),
/// computes the per-pixel stain concentrations via:
///
///     C = SDA_pixels @ (W⁻¹)ᵀ
///
/// This follows from the Beer-Lambert model: OD = W @ c  ⟹  c = W⁻¹ @ OD.
///
/// # Arguments
///
/// * `im_sda`        – SDA image with shape (H, W, 3).
/// * `stain_matrix`  – 3×3 matrix whose **columns** are normalised stain vectors
///                      (as returned by `separate_stains_macenko_pca`).
///
/// # Returns
///
/// A `(H, W, 3)` array where channel *i* holds the concentration of stain *i*.
pub fn color_deconvolution<F: MacenkoFloat>(
    im_sda: ArrayView3<F>,
    stain_matrix: ArrayView2<F>,
) -> Result<Array3<F>, String> {
    let shape = im_sda.shape();
    let (h, w, c) = (shape[0], shape[1], shape[2]);

    if c != 3 {
        return Err(format!("Image must have 3 channels, got {c}"));
    }
    if stain_matrix.shape() != [3, 3] {
        return Err(format!(
            "Stain matrix must be 3×3, got {:?}",
            stain_matrix.shape()
        ));
    }

    // Invert the stain matrix: W⁻¹
    let w_inv = stain_matrix
        .to_owned()
        .inv()
        .map_err(|e| format!("Failed to invert stain matrix: {e:?}"))?;

    // Flatten image to (N, 3) where N = H*W
    let n_pixels = h * w;
    let sda_flat = im_sda
        .to_shape((n_pixels, 3))
        .map_err(|e| format!("Failed to reshape SDA image: {e:?}"))?;

    // concentrations = sda_flat @ W⁻ᵀ
    let w_inv_t = w_inv.t().to_owned();
    let result = sda_flat.dot(&w_inv_t);

    // Reshape back to (H, W, 3)
    result
        .into_shape_with_order((h, w, 3))
        .map_err(|e| format!("Failed to reshape result: {e:?}"))
}

/// Convenience wrapper: convert RGB → SDA, then deconvolve into stain concentrations.
///
/// # Arguments
///
/// * `im_rgb`        – RGB image with shape (H, W, 3), values in [0, 255].
/// * `stain_matrix`  – 3×3 stain matrix (columns = stain vectors).
/// * `bg_int`        – Optional per-channel background intensity for the SDA transform.
///
/// # Returns
///
/// A `(H, W, 3)` concentration image.
pub fn rgb_color_deconvolution<F: MacenkoFloat>(
    im_rgb: ArrayView3<F>,
    stain_matrix: ArrayView2<F>,
    bg_int: Option<&[F]>,
) -> Result<Array3<F>, String> {
    let im_sda = rgb_to_sda(im_rgb, bg_int, None)
        .map_err(|e| format!("RGB→SDA conversion failed: {e:?}"))?;
    color_deconvolution(im_sda.view(), stain_matrix)
}

/// Reconstruct an RGB image from stain concentrations and a stain matrix.
///
/// Inverts the deconvolution process:
///
///     SDA = concentrations @ Wᵀ
///     RGB = bg_int × exp(−SDA × ln(bg_int) / 255)
///
/// # Arguments
///
/// * `concentrations` – (H, W, 3) stain concentration image.
/// * `stain_matrix`   – 3×3 stain matrix (columns = stain vectors).
/// * `bg_int`         – Background intensity used during the original SDA transform
///                       (scalar, applied uniformly). Defaults to 256.
///
/// # Returns
///
/// A `(H, W, 3)` reconstructed RGB image with values in [0, bg_int].
pub fn reconstruct_rgb<F: MacenkoFloat>(
    concentrations: ArrayView3<F>,
    stain_matrix: ArrayView2<F>,
    bg_int: Option<F>,
) -> Result<Array3<F>, String> {
    let shape = concentrations.shape();
    let (h, w, c) = (shape[0], shape[1], shape[2]);

    if c != 3 {
        return Err(format!("Concentrations must have 3 channels, got {c}"));
    }
    if stain_matrix.shape() != [3, 3] {
        return Err(format!(
            "Stain matrix must be 3×3, got {:?}",
            stain_matrix.shape()
        ));
    }

    let bg = bg_int.unwrap_or_else(|| F::from(256.0).unwrap());
    let ln_bg = Float::ln(bg);
    let scale = ln_bg / F::from(255.0).unwrap();
    let zero = F::zero();

    let n_pixels = h * w;

    // Flatten concentrations to (N, 3)
    let conc_flat = concentrations
        .to_shape((n_pixels, 3))
        .map_err(|e| format!("Failed to reshape concentrations: {e:?}"))?;

    // SDA = concentrations @ Wᵀ
    let w_t = stain_matrix.t().to_owned();
    let sda_flat = conc_flat.dot(&w_t);

    // Convert SDA back to RGB: rgb_ch = bg * exp(-sda_ch * ln(bg) / 255)
    let mut rgb_data: Vec<F> = vec![zero; n_pixels * 3];
    rgb_data
        .par_chunks_mut(3)
        .enumerate()
        .for_each(|(i, pixel)| {
            for ch in 0..3 {
                let sda_val = sda_flat[[i, ch]];
                let rgb_val = bg * Float::exp(-sda_val * scale);
                // Clamp to [0, bg]
                pixel[ch] = Float::min(Float::max(rgb_val, zero), bg);
            }
        });

    ndarray::Array3::from_shape_vec((h, w, 3), rgb_data)
        .map_err(|e| format!("Failed to reshape RGB result: {e:?}"))
}
