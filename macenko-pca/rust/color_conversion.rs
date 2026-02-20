use crate::float_trait::MacenkoFloat;
use ndarray::parallel::prelude::*;
use ndarray::{Array, ArrayView, Dimension, RemoveAxis, ShapeError};
use num_traits::Float;

pub fn rgb_to_sda<F, D>(
    rgb_im: ArrayView<F, D>,
    bg_int: Option<&[F]>,
    allow_negative: Option<bool>,
) -> Result<Array<F, D>, ShapeError>
where
    F: MacenkoFloat,
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
    let default_bg = F::from(256.0).unwrap();
    let bg_int = match bg_int {
        Some(arr) if arr.len() == 1 => vec![arr[0]; n_channels],
        Some(arr) if arr.len() == n_channels => arr.to_vec(),
        None => vec![default_bg; n_channels],
        _ => return Err(ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape)),
    };

    if bg_int.iter().all(|&v| v == bg_int[0]) {
        rgb_to_sda_intensity(rgb_view, bg_int[0], allow_negative, is_matrix)
    } else {
        rgb_to_sda_per_channel(rgb_view, &bg_int, allow_negative, is_matrix)
    }
}

fn rgb_to_sda_intensity<F, D>(
    rgb: ArrayView<F, D>,
    bg_int: F,
    allow_negative: bool,
    is_matrix: bool,
) -> Result<Array<F, D>, ShapeError>
where
    F: MacenkoFloat,
    D: Dimension,
{
    let ln_bg = Float::ln(bg_int);
    let scale = F::from(255.0).unwrap() / ln_bg;
    let epsilon = F::from(1e-10).unwrap();
    let mut im_sda = rgb.to_owned();

    im_sda.par_iter_mut().for_each(|x| {
        let val = -(Float::ln(F::max(*x, epsilon) / bg_int)) * scale;
        *x = if allow_negative {
            val
        } else {
            F::max(val, F::zero())
        };
    });

    if is_matrix {
        Ok(im_sda.reversed_axes())
    } else {
        Ok(im_sda)
    }
}

fn rgb_to_sda_per_channel<F, D>(
    rgb: ArrayView<F, D>,
    bg_int: &[F],
    allow_negative: bool,
    is_matrix: bool,
) -> Result<Array<F, D>, ShapeError>
where
    F: MacenkoFloat,
    D: Dimension + RemoveAxis,
{
    let mut im_sda = rgb.to_owned();
    let scale_factor = F::from(255.0).unwrap();
    let epsilon = F::from(1e-10).unwrap();

    let n_ch = bg_int.len();
    im_sda.outer_iter_mut().into_par_iter().for_each(|mut row| {
        for (i, x) in row.iter_mut().enumerate() {
            let chan = i % n_ch;
            let bg = bg_int[chan];
            let ln_bg = Float::ln(bg);
            let scale = scale_factor / ln_bg;
            let val = -(Float::ln(F::max(*x, epsilon) / bg)) * scale;
            *x = if allow_negative {
                val
            } else {
                F::max(val, F::zero())
            };
        }
    });

    if is_matrix {
        Ok(im_sda.reversed_axes())
    } else {
        Ok(im_sda)
    }
}
