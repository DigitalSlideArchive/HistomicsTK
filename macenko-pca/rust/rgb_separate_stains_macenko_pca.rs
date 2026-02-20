use crate::color_conversion::rgb_to_sda;
use crate::float_trait::MacenkoFloat;
use crate::separate_stains_macenko_pca::separate_stains_macenko_pca;
use ndarray::{Array2, ArrayView2, ArrayView3};

/// Compute the stain matrix for color deconvolution with the "Macenko" method from an RGB image or matrix.
pub fn rgb_separate_stains_macenko_pca<F: MacenkoFloat>(
    im_rgb: ArrayView3<F>,
    bg_int: Option<&[F]>,
    minimum_magnitude: F,
    min_angle_percentile: F,
    max_angle_percentile: F,
    mask_out: Option<ArrayView2<bool>>,
) -> Result<Array2<F>, ndarray::ShapeError> {
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
