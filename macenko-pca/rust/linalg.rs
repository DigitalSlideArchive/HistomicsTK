use crate::float_trait::MacenkoFloat;
use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::SVD;
use num_traits::Float;
use rayon::prelude::*;

/// Take a matrix m (probably 3xN) and return the 3x3 matrix of principal components (via SVD)
pub fn get_principal_components<F: MacenkoFloat>(m: &Array2<F>) -> Array2<F> {
    // SVD: m = U * S * V^T, return U (principal components)
    // Only compute U, not V^T
    let (u_opt, _, _) = m.svd(true, false).expect("SVD failed");
    u_opt.expect("U not computed")
}

/// Get the magnitude of each column vector in a matrix
pub fn magnitude<F: MacenkoFloat>(m: &Array2<F>) -> Array1<F> {
    // Parallelize across columns for large matrices
    let v: Vec<F> = (0..m.ncols())
        .into_par_iter()
        .map(|i| Float::sqrt(m.column(i).dot(&m.column(i))))
        .collect();
    Array1::from(v)
}

/// Normalize each column vector in a matrix
pub fn normalize<F: MacenkoFloat>(m: &Array2<F>) -> Array2<F> {
    let mag = magnitude(m);
    let nrows = m.nrows();
    let ncols = m.ncols();
    let mut result = Array2::<F>::zeros((nrows, ncols));

    result
        .axis_iter_mut(Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut col)| {
            let norm = mag[i];
            if norm != F::zero() {
                for (j, v) in col.iter_mut().enumerate() {
                    *v = m[[j, i]] / norm;
                }
            } else {
                for v in col.iter_mut() {
                    *v = F::zero();
                }
            }
        });

    result
}
