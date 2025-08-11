use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyReadonlyArray3}; // <-- add IntoPyArray here
use pyo3::PyResult;
use pyo3::Python;
use pyo3::prelude::*;
use pyo3::pyfunction;

mod color_conversion;
mod complement_stain_matrix;
mod linalg;
mod rgb_separate_stains_macenko_pca;
mod separate_stains_macenko_pca;
mod utils;

#[pyfunction]
fn py_rgb_separate_stains_macenko_pca<'py>(
    py: Python<'py>,
    im_rgb: PyReadonlyArray3<'py, f64>,
    bg_int: Option<Vec<f64>>,
    minimum_magnitude: Option<f64>,
    min_angle_percentile: Option<f64>,
    max_angle_percentile: Option<f64>,
    mask_out: Option<PyReadonlyArray2<'py, bool>>,
) -> PyResult<pyo3::Bound<'py, PyArray2<f64>>> {
    let arr = im_rgb.as_array();
    let mask_ref = mask_out.as_ref();
    let mask = mask_ref.map(|m| m.as_array());

    let bg_ref = bg_int.as_ref().map(|v| v.as_slice());

    let result = rgb_separate_stains_macenko_pca::rgb_separate_stains_macenko_pca(
        arr,
        bg_ref,
        minimum_magnitude.unwrap_or(16.0),
        min_angle_percentile.unwrap_or(0.01),
        max_angle_percentile.unwrap_or(0.99),
        mask,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{:?}", e)))?;

    Ok(result.into_pyarray(py))
}

#[pymodule]
fn _rust(_py: Python, m: pyo3::Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_rgb_separate_stains_macenko_pca, &m)?)?;
    Ok(())
}
