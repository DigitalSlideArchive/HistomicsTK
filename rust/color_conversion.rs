use ndarray::parallel::prelude::*;
use ndarray::{Array, ArrayView, Dimension, RemoveAxis, ShapeError};

pub fn rgb_to_sda<D>(
    rgb_im: ArrayView<f64, D>,
    bg_int: Option<&[f64]>,
    allow_negative: Option<bool>,
) -> Result<Array<f64, D>, ShapeError>
where
    D: Dimension + RemoveAxis,
{
    let allow_negative = allow_negative.unwrap_or(false);

    let is_matrix = rgb_im.ndim() == 2;
    let rgb_view = if is_matrix {
        rgb_im.reversed_axes()
    } else if rgb_im.ndim() != 3 {
        return Err(ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape));
    } else {
        rgb_im
    };

    let n_channels = rgb_view.shape()[rgb_view.ndim() - 1];
    let bg_int = match bg_int {
        Some(arr) if arr.len() == 1 => vec![arr[0]; n_channels],
        Some(arr) if arr.len() == n_channels => arr.to_vec(),
        None => vec![256.0; n_channels],
        _ => return Err(ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape)),
    };

    if bg_int.iter().all(|&v| v == bg_int[0]) {
        rgb_to_sda_intensity(rgb_view, bg_int[0], allow_negative, is_matrix)
    } else {
        rgb_to_sda_per_channel(rgb_view, &bg_int, allow_negative, is_matrix)
    }
}
fn rgb_to_sda_intensity<D>(
    rgb: ArrayView<f64, D>,
    bg_int: f64,
    allow_negative: bool,
    is_matrix: bool,
) -> Result<Array<f64, D>, ShapeError>
where
    D: Dimension,
{
    let ln_bg = bg_int.ln();
    let scale = 255.0 / ln_bg;
    let mut im_sda = rgb.to_owned();

    im_sda.par_iter_mut().for_each(|x| {
        let val = -((f64::max(*x, 1e-10) / bg_int).ln()) * scale;
        *x = if allow_negative {
            val
        } else {
            f64::max(val, 0.0)
        };
    });

    if is_matrix {
        Ok(im_sda.reversed_axes())
    } else {
        Ok(im_sda)
    }
}

fn rgb_to_sda_per_channel<D>(
    rgb: ArrayView<f64, D>,
    bg_int: &[f64],
    allow_negative: bool,
    is_matrix: bool,
) -> Result<Array<f64, D>, ShapeError>
where
    D: Dimension + RemoveAxis,
{
    let mut im_sda = rgb.to_owned();

    im_sda.outer_iter_mut().into_par_iter().for_each(|mut row| {
        for (chan, x) in row.iter_mut().enumerate() {
            let bg = bg_int[chan];
            let ln_bg = bg.ln();
            let scale = 255.0 / ln_bg;
            let val = -((f64::max(*x, 1e-10) / bg).ln()) * scale;
            *x = if allow_negative {
                val
            } else {
                f64::max(val, 0.0)
            };
        }
    });

    if is_matrix {
        Ok(im_sda.reversed_axes())
    } else {
        Ok(im_sda)
    }
}
