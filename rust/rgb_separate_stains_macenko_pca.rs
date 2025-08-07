use crate::color_conversion::rgb_to_sda;
use crate::separate_stains_macenko_pca::separate_stains_macenko_pca;
use ndarray::{Array2, ArrayView2, ArrayView3};

/// Compute the stain matrix for color deconvolution with the "Macenko" method from an RGB image or matrix.
pub fn rgb_separate_stains_macenko_pca(
    im_rgb: ArrayView3<f64>,
    bg_int: Option<&[f64]>,
    minimum_magnitude: f64,
    min_angle_percentile: f64,
    max_angle_percentile: f64,
    mask_out: Option<ArrayView2<bool>>,
) -> Result<Array2<f64>, ndarray::ShapeError> {
    // Convert RGB to SDA space, move im_rgb to this function for memory efficiency
    let im_sda = rgb_to_sda(im_rgb, bg_int, None)?;
    let mask = match mask_out {
        Some(mask) => Some(mask),
        None => None,
    };

    // Compute stain matrix
    Ok(separate_stains_macenko_pca(
        im_sda,
        minimum_magnitude,
        min_angle_percentile,
        max_angle_percentile,
        mask.as_ref(),
    ))
}
