from .sda_to_rgb import sda_to_rgb


def od_to_rgb(im_od):
    """Transforms input optical density image `im_od` into RGB space

    Parameters
    ----------
    im_od : array_like
        A floating-point image of optical density values obtained
        from rgb_to_od.

    Returns
    -------
    im_rgb : array_like
        A floating-point multi-channel image with intensity
        values in the range [0, 255].

    See Also
    --------
    histomicstk.preprocessing.color_conversion.rgb_to_od,
    histomicstk.preprocessing.color_conversion.sda_to_rgb,
    histomicstk.preprocessing.color_deconvolution.color_deconvolution,
    histomicstk.preprocessing.color_deconvolution.color_convolution

    """

    return sda_to_rgb(im_od, None)  # compatibility mode
