use crate::complement_stain_matrix::complement_stain_matrix;
use crate::linalg::{get_principal_components, magnitude, normalize};
use crate::utils::{convert_image_to_matrix, exclude_nonfinite};
use ndarray::{Array1, Array2, Array3, ArrayView2, Axis, s};
use rayon::prelude::*; // Add this at the top

/// Compute the stain matrix for color deconvolution with the Macenko method.
pub fn separate_stains_macenko_pca(
    im_sda: Array3<f64>,
    minimum_magnitude: f64,
    min_angle_percentile: f64,
    max_angle_percentile: f64,
    mask_out: Option<&ArrayView2<bool>>,
) -> Array2<f64> {
    // Convert image to matrix
    let mut m = convert_image_to_matrix(im_sda);
    // Mask out irrelevant values
    if let Some(mask) = mask_out {
        // mask is H×W; convert it to a flat list of pixel-indices where mask==false
        let mask_indices: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter(|&(_, &is_masked)| !is_masked)
            .map(|(flat_idx, _)| flat_idx)
            .collect();
        m = select_columns_parallel(&m, &mask_indices);
    }

    // Exclude NaNs and infinities
    m = exclude_nonfinite(m);

    // Principal components matrix
    let pcs = get_principal_components(&m);

    // Project input pixels into the PCA plane
    let proj = pcs.t().slice(s![..-1, ..]).dot(&m);
    drop(m);

    // Pixels above the magnitude threshold
    let mag = magnitude(&proj);
    let filt_indices: Vec<usize> = mag
        .iter()
        .enumerate()
        .filter(|(_, v)| **v > minimum_magnitude)
        .map(|(i, _)| i)
        .collect();
    drop(mag);
    let filt = select_columns_parallel(&proj, &filt_indices);
    drop(proj);
    // The "angles"
    let angles = get_angles(&filt);

    // The stain vectors
    let percentiles = [min_angle_percentile, max_angle_percentile];
    let vectors: Vec<Array1<f64>> = percentiles
        .par_iter()
        .map(|&p| get_percentile_vector(&pcs, &filt, &angles, p))
        .collect();
    let min_v = &vectors[0];
    let max_v = &vectors[1];
    drop(pcs);
    drop(filt);
    drop(angles);

    // The stain matrix
    let mut stains = ndarray::Array2::<f64>::zeros((min_v.len(), 2));
    stains
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            row[0] = min_v[i];
            row[1] = max_v[i];
        });
    complement_stain_matrix(&normalize(&stains))
}

/// faster, lower‐overhead version of select_columns
fn select_columns_parallel(arr: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
    let nrows = arr.nrows();
    let ncols = indices.len();
    if ncols == 0 {
        return Array2::zeros((nrows, 0));
    }

    // build one flat Vec in parallel
    let new_data: Vec<f64> = (0..nrows * ncols)
        .into_par_iter()
        .map(|flat| {
            let row = flat / ncols;
            let col = indices[flat % ncols];
            arr[[row, col]]
        })
        .collect();

    Array2::from_shape_vec((nrows, ncols), new_data).expect("Shape conversion should not fail")
}

/// compute the same “angles” but without a full normalize() allocation
fn get_angles(mat: &Array2<f64>) -> Array1<f64> {
    let n = mat.ncols();
    let r0 = mat.row(0);
    let r1 = mat.row(1);
    let mut ang = Array1::zeros(n);
    for i in 0..n {
        let x = r0[i];
        let y = r1[i];
        let mag = (x * x + y * y).sqrt();
        ang[i] = if mag != 0.0 {
            let xn = x / mag;
            let yn = y / mag;
            (1.0 - yn) * xn.signum()
        } else {
            0.0
        };
    }
    ang
}

fn argpercentile(arr: &Array1<f64>, p: f64) -> usize {
    let size = arr.len();
    let i = ((p * size as f64 + 0.5).min((size - 1) as f64)) as usize;
    let mut indices: Vec<usize> = (0..size).collect();
    indices.select_nth_unstable_by(i, |&a, &b| arr[a].partial_cmp(&arr[b]).unwrap());
    indices[i]
}

fn get_percentile_vector(
    pcs: &Array2<f64>,
    filt: &Array2<f64>,
    angles: &Array1<f64>,
    percentile: f64,
) -> Array1<f64> {
    let idx = argpercentile(angles, percentile);
    pcs.slice(s![.., ..pcs.ncols() - 1]).dot(&filt.column(idx))
}
