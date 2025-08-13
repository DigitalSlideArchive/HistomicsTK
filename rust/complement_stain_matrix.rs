use ndarray::prelude::*;
use ndarray::{Array1, Array2, ArrayView1};
use ndarray_linalg::Norm;

/// Generates a complemented stain matrix.
/// Replaces right-most column with normalized cross-product of first two columns.
pub fn complement_stain_matrix(w: &Array2<f64>) -> Array2<f64> {
    let stain0 = w.column(0);
    let stain1 = w.column(1);
    let stain2 = cross_product(&stain0, &stain1);
    let stain2_norm = stain2.mapv(|x| x) / stain2.norm_l2();
    let mut result = Array2::<f64>::zeros((3, 3));
    result.column_mut(0).assign(&stain0);
    result.column_mut(1).assign(&stain1);
    result.column_mut(2).assign(&stain2_norm);
    result
}

/// Compute cross product of two 3-element vectors
fn cross_product(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
    array![
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ]
}
