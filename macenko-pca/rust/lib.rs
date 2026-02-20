use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyArrayDyn, PyReadonlyArray2, PyReadonlyArray3,
    PyReadonlyArrayDyn,
};
use pyo3::PyResult;
use pyo3::Python;
use pyo3::prelude::*;
use pyo3::pyfunction;

mod color_conversion;
mod color_deconvolution;
mod complement_stain_matrix;
mod float_trait;
mod linalg;
mod rgb_separate_stains_macenko_pca;
mod separate_stains_macenko_pca;
mod utils;

// ---------------------------------------------------------------------------
// Macro to stamp out typed PyO3 function variants for each float precision.
//
// Each invocation generates three #[pyfunction]s:
//   - {prefix}_rgb_separate_stains_macenko_pca_{suffix}
//   - {prefix}_separate_stains_macenko_pca_{suffix}
//   - {prefix}_rgb_to_sda_{suffix}
//   - {prefix}_color_deconvolution_{suffix}
//   - {prefix}_rgb_color_deconvolution_{suffix}
//   - {prefix}_reconstruct_rgb_{suffix}
// ---------------------------------------------------------------------------
macro_rules! define_typed_pyfunctions {
    ($suffix:ident, $f:ty) => {
        ::paste::paste! {

        #[pyfunction]
        #[pyo3(signature = (im_rgb, bg_int=None, minimum_magnitude=None, min_angle_percentile=None, max_angle_percentile=None, mask_out=None))]
        fn [<py_rgb_separate_stains_macenko_pca_ $suffix>]<'py>(
            py: Python<'py>,
            im_rgb: PyReadonlyArray3<'py, $f>,
            bg_int: Option<Vec<f64>>,
            minimum_magnitude: Option<f64>,
            min_angle_percentile: Option<f64>,
            max_angle_percentile: Option<f64>,
            mask_out: Option<PyReadonlyArray2<'py, bool>>,
        ) -> PyResult<pyo3::Bound<'py, PyArray2<$f>>> {
            let arr = im_rgb.as_array();
            let mask_ref = mask_out.as_ref();
            let mask = mask_ref.map(|m| m.as_array());

            let bg: Option<Vec<$f>> = bg_int.map(|v| v.iter().map(|&x| x as $f).collect());
            let bg_ref = bg.as_ref().map(|v| v.as_slice());

            let result =
                rgb_separate_stains_macenko_pca::rgb_separate_stains_macenko_pca(
                    arr,
                    bg_ref,
                    minimum_magnitude.unwrap_or(16.0) as $f,
                    min_angle_percentile.unwrap_or(0.01) as $f,
                    max_angle_percentile.unwrap_or(0.99) as $f,
                    mask,
                )
                .map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("{:?}", e))
                })?;

            Ok(result.into_pyarray(py))
        }

        #[pyfunction]
        #[pyo3(signature = (im_sda, minimum_magnitude=None, min_angle_percentile=None, max_angle_percentile=None, mask_out=None))]
        fn [<py_separate_stains_macenko_pca_ $suffix>]<'py>(
            py: Python<'py>,
            im_sda: PyReadonlyArrayDyn<'py, $f>,
            minimum_magnitude: Option<f64>,
            min_angle_percentile: Option<f64>,
            max_angle_percentile: Option<f64>,
            mask_out: Option<PyReadonlyArray2<'py, bool>>,
        ) -> PyResult<pyo3::Bound<'py, PyArray2<$f>>> {
            use ndarray::Ix3;
            let arr = im_sda.as_array();
            let arr3 = if arr.ndim() == 3 {
                arr.into_dimensionality::<Ix3>().unwrap()
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "im_sda must be 3D",
                ));
            };
            let mask_ref = mask_out.as_ref();
            let mask = mask_ref.map(|m| m.as_array());
            let result =
                crate::separate_stains_macenko_pca::separate_stains_macenko_pca(
                    arr3.to_owned(),
                    minimum_magnitude.unwrap_or(16.0) as $f,
                    min_angle_percentile.unwrap_or(0.01) as $f,
                    max_angle_percentile.unwrap_or(0.99) as $f,
                    mask.as_ref(),
                );
            Ok(result.into_pyarray(py))
        }

        #[pyfunction]
        #[pyo3(signature = (im_rgb, bg_int=None, allow_negative=None))]
        fn [<py_rgb_to_sda_ $suffix>]<'py>(
            py: Python<'py>,
            im_rgb: PyReadonlyArrayDyn<'py, $f>,
            bg_int: Option<Vec<f64>>,
            allow_negative: Option<bool>,
        ) -> PyResult<pyo3::Bound<'py, PyArrayDyn<$f>>> {
            let arr = im_rgb.as_array();
            let bg: Option<Vec<$f>> = bg_int.map(|v| v.iter().map(|&x| x as $f).collect());
            let bg_ref = bg.as_ref().map(|v| v.as_slice());
            let result =
                color_conversion::rgb_to_sda(arr, bg_ref, allow_negative)
                    .map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!("{:?}", e))
                    })?;
            Ok(result.into_pyarray(py))
        }

        #[pyfunction]
        #[pyo3(signature = (im_sda, stain_matrix))]
        fn [<py_color_deconvolution_ $suffix>]<'py>(
            py: Python<'py>,
            im_sda: PyReadonlyArray3<'py, $f>,
            stain_matrix: PyReadonlyArray2<'py, $f>,
        ) -> PyResult<pyo3::Bound<'py, PyArray3<$f>>> {
            let arr = im_sda.as_array();
            let w = stain_matrix.as_array();
            let result =
                color_deconvolution::color_deconvolution(arr, w)
                    .map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(e)
                    })?;
            Ok(result.into_pyarray(py))
        }

        #[pyfunction]
        #[pyo3(signature = (im_rgb, stain_matrix, bg_int=None))]
        fn [<py_rgb_color_deconvolution_ $suffix>]<'py>(
            py: Python<'py>,
            im_rgb: PyReadonlyArray3<'py, $f>,
            stain_matrix: PyReadonlyArray2<'py, $f>,
            bg_int: Option<Vec<f64>>,
        ) -> PyResult<pyo3::Bound<'py, PyArray3<$f>>> {
            let arr = im_rgb.as_array();
            let w = stain_matrix.as_array();
            let bg: Option<Vec<$f>> = bg_int.map(|v| v.iter().map(|&x| x as $f).collect());
            let bg_ref = bg.as_ref().map(|v| v.as_slice());
            let result =
                color_deconvolution::rgb_color_deconvolution(arr, w, bg_ref)
                    .map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(e)
                    })?;
            Ok(result.into_pyarray(py))
        }

        #[pyfunction]
        #[pyo3(signature = (concentrations, stain_matrix, bg_int=None))]
        fn [<py_reconstruct_rgb_ $suffix>]<'py>(
            py: Python<'py>,
            concentrations: PyReadonlyArray3<'py, $f>,
            stain_matrix: PyReadonlyArray2<'py, $f>,
            bg_int: Option<f64>,
        ) -> PyResult<pyo3::Bound<'py, PyArray3<$f>>> {
            let arr = concentrations.as_array();
            let w = stain_matrix.as_array();
            let bg: Option<$f> = bg_int.map(|v| v as $f);
            let result =
                color_deconvolution::reconstruct_rgb(arr, w, bg)
                    .map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(e)
                    })?;
            Ok(result.into_pyarray(py))
        }

        } // paste!
    };
}

// Stamp out the concrete f32 and f64 variants.
define_typed_pyfunctions!(f32, f32);
define_typed_pyfunctions!(f64, f64);

// ---------------------------------------------------------------------------
// PyO3 module — register every typed variant.
// ---------------------------------------------------------------------------
#[pymodule]
fn _rust(_py: Python, m: pyo3::Bound<'_, PyModule>) -> PyResult<()> {
    // f64
    m.add_function(wrap_pyfunction!(
        py_rgb_separate_stains_macenko_pca_f64,
        &m
    )?)?;
    m.add_function(wrap_pyfunction!(py_separate_stains_macenko_pca_f64, &m)?)?;
    m.add_function(wrap_pyfunction!(py_rgb_to_sda_f64, &m)?)?;
    m.add_function(wrap_pyfunction!(py_color_deconvolution_f64, &m)?)?;
    m.add_function(wrap_pyfunction!(py_rgb_color_deconvolution_f64, &m)?)?;
    m.add_function(wrap_pyfunction!(py_reconstruct_rgb_f64, &m)?)?;
    // f32
    m.add_function(wrap_pyfunction!(
        py_rgb_separate_stains_macenko_pca_f32,
        &m
    )?)?;
    m.add_function(wrap_pyfunction!(py_separate_stains_macenko_pca_f32, &m)?)?;
    m.add_function(wrap_pyfunction!(py_rgb_to_sda_f32, &m)?)?;
    m.add_function(wrap_pyfunction!(py_color_deconvolution_f32, &m)?)?;
    m.add_function(wrap_pyfunction!(py_rgb_color_deconvolution_f32, &m)?)?;
    m.add_function(wrap_pyfunction!(py_reconstruct_rgb_f32, &m)?)?;
    Ok(())
}
