use cauchy::Scalar;
use lax::Lapack;
use num_traits::Float;
use numpy::Element;

/// Supertrait combining all bounds needed by the generic Macenko PCA pipeline.
///
/// Implemented for `f32` and `f64` only — these are the two types supported by
/// LAPACK (via `ndarray-linalg`) and by the NumPy ↔ Rust bridge.
///
/// `f16` is intentionally excluded: there is no f16 LAPACK, so the Python layer
/// upcasts float16 inputs to float32 before crossing into Rust.
pub trait MacenkoFloat:
    Float + Scalar<Real = Self> + Lapack + Element + Send + Sync + 'static
{
}

impl MacenkoFloat for f32 {}
impl MacenkoFloat for f64 {}
