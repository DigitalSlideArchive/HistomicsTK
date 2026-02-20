use crate::float_trait::MacenkoFloat;
use ndarray::{Array2, Array3, Axis};

pub fn convert_image_to_matrix<T>(im: Array3<T>) -> Array2<T>
where
    T: Clone,
{
    let shape = im.shape();
    let (m, n, c) = (shape[0], shape[1], shape[2]);
    im.into_shape_with_order((m * n, c))
        .unwrap()
        .reversed_axes()
}

pub fn exclude_nonfinite<F: MacenkoFloat>(m: Array2<F>) -> Array2<F> {
    let valid_indices: Vec<usize> = m
        .columns()
        .into_iter()
        .enumerate()
        .filter_map(|(i, col)| {
            if col.iter().all(|&x| x.is_finite()) {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    // If all columns were valid, return the original array, avoiding a copy.
    if valid_indices.len() == m.ncols() {
        return m;
    }
    m.select(Axis(1), &valid_indices)
}
