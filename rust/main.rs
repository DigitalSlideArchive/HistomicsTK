mod color_conversion;
mod complement_stain_matrix;
mod linalg;
mod rgb_separate_stains_macenko_pca;
mod separate_stains_macenko_pca;
mod utils;
use ndarray::{Array3, s};
use ndarray_npy::read_npy;
use std::time::Instant;

fn main() {
    // Read the .npy file
    let arr: Array3<f64> = read_npy("test_image.npy").expect("Failed to load .npy file");

    // Profile the deconvolution
    let start = Instant::now();
    let result = rgb_separate_stains_macenko_pca::rgb_separate_stains_macenko_pca(
        arr.view(),
        None,
        16.0,
        0.01,
        0.99,
        None,
    )
    .expect("Deconvolution failed");
    println!("Deconvolution took {:?}", start.elapsed());

    // Print part of the result
    println!("{:?}", result.slice(s![.., ..3]));
}
