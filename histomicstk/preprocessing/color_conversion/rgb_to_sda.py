import numpy as np


def rgb_to_sda(im_rgb, I_0, allow_negatives=False):
    """Transform input RGB image or matrix `im_rgb` into SDA (stain
    darkness) space for color deconvolution.

    Parameters
    ----------
    im_rgb : array_like
        Image (MxNx3) or matrix (3xN) of pixels

    I_0 : float or array_like
        Background intensity, either per-channel or for all channels

    allow_negatives : bool
        If False, would-be negative values in the output are clipped to 0

    Returns
    -------
    im_sda : array_like
        Shaped like `im_rgb`, with output values 0..255 where `im_rgb` >= 1

    Note
    ----
    For compatibility purposes, passing I_0=None invokes the behavior of
    rgb_to_od.

    See Also
    --------
    histomicstk.preprocessing.color_conversion.sda_to_rgb,
    histomicstk.preprocessing.color_conversion.rgb_to_od,
    histomicstk.preprocessing.color_deconvolution.color_deconvolution,
    histomicstk.preprocessing.color_deconvolution.color_convolution

    """
    is_matrix = im_rgb.ndim == 2
    if is_matrix:
        im_rgb = im_rgb.T

    if I_0 is None:  # rgb_to_od compatibility
        im_rgb = im_rgb.astype(float) + 1
        I_0 = 256

    im_rgb = np.maximum(im_rgb, 1e-10)

    im_sda = -np.log(im_rgb / (1. * I_0)) * 255 / np.log(I_0)
    if not allow_negatives:
        im_sda = np.maximum(im_sda, 0)
    return im_sda.T if is_matrix else im_sda
