import numpy as np
from histomicstk._rust import (  # pylint: disable=no-name-in-module
    py_rgb_separate_stains_macenko_pca,
)
from ..color_conversion import rgb_to_sda
from .separate_stains_macenko_pca import separate_stains_macenko_pca


def rgb_separate_stains_macenko_pca_rs(
    im_rgb,
    I_0,
    minimum_magnitude=16,
    min_angle_percentile=0.01,
    max_angle_percentile=0.99,
    mask_out=None,
):
    """
    Rust-accelerated version of rgb_separate_stains_macenko_pca.
    """
    im = np.asarray(im_rgb, dtype=np.float64, order='C')
    mask = None
    if mask_out is not None:
        mask = np.ascontiguousarray(mask_out, dtype=bool)
    # Prepare background intensity argument
    if I_0 is None:
        im = im + 1.0
        bg = None
    elif np.isscalar(I_0):
        bg = [float(I_0)]
    else:
        bg = np.asarray(I_0, dtype=np.float64).ravel().tolist()
    return py_rgb_separate_stains_macenko_pca(
        im, bg, minimum_magnitude, min_angle_percentile, max_angle_percentile, mask,
    )


def rgb_separate_stains_macenko_pca(im_rgb, I_0, *args, **kwargs):
    """Compute the stain matrix for color deconvolution with the "Macenko"
    method from an RGB image or matrix.

    Parameters
    ----------
    im_rgb : array_like
        Image (MxNx3) or matrix (3xN) in RGB space for which to compute the
        stain matrix.
    I_0 : float or array_like
        Per-channel background intensities, or one intensity to use for all
        channels if a float.

    Returns
    -------
    w : array_like
        A 3x3 matrix of stain column vectors, in SDA space

    Note
    ----
    For additional input arguments and documentation, please see
    histomicstk.preprocessing.color_deconvolution.separate_stains_macenko_pca.
    im_sda is computed and passed as part of this routine.

    See Also
    --------
    histomicstk.preprocessing.color_deconvolution.separate_stains_macenko_pca

    """
    im_sda = rgb_to_sda(im_rgb, I_0)
    return separate_stains_macenko_pca(im_sda, *args, **kwargs)
