use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::SVD;
use rayon::prelude::*;

/// Take a matrix m (probably 3xN) and return the 3x3 matrix of principal components (via SVD)
pub fn get_principal_components(m: &Array2<f64>) -> Array2<f64> {
    // SVD: m = U * S * V^T, return U (principal components)
    // Only compute U, not V^T
    let (u_opt, _, _) = m.svd(true, false).expect("SVD failed");
    u_opt.expect("U not computed")
}

/// Get the magnitude of each column vector in a matrix
pub fn magnitude(m: &Array2<f64>) -> Array1<f64> {
    // Parallelize across columns for large matrices
    let v: Vec<f64> = (0..m.ncols())
        .into_par_iter()
        .map(|i| m.column(i).dot(&m.column(i)).sqrt())
        .collect();
    Array1::from(v)
}

/// Normalize each column vector in a matrix
pub fn normalize(m: &Array2<f64>) -> Array2<f64> {
    let mag = magnitude(m);
    let nrows = m.nrows();
    let ncols = m.ncols();
    let mut result = Array2::<f64>::zeros((nrows, ncols));

    result
        .axis_iter_mut(Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut col)| {
            let norm = mag[i];
            if norm != 0.0 {
                for (j, v) in col.iter_mut().enumerate() {
                    *v = m[[j, i]] / norm;
                }
            } else {
                for v in col.iter_mut() {
                    *v = 0.0;
                }
            }
        });

    result
}
