from .rgb_to_sda import rgb_to_sda


def rgb_to_od(im_rgb):
    """Transforms input RGB image `im_rgb` into optical density space
    for color deconvolution.

    Parameters
    ----------
    im_rgb : array_like
        A floating-point RGB image with intensity ranges of [0, 255].

    Returns
    -------
    im_od : array_like
        A floating-point image of corresponding optical density values.


    See Also
    --------
    histomicstk.preprocessing.color_conversion.od_to_rgb,
    histomicstk.preprocessing.color_conversion.rgb_to_sda,
    histomicstk.preprocessing.color_deconvolution.color_deconvolution,
    histomicstk.preprocessing.color_deconvolution.color_convolution

    """

    return rgb_to_sda(im_rgb, None)  # Compatibility mode
