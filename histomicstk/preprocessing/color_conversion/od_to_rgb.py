import numpy as np


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
    histomicstk.preprocessing.color_deconvolution.ColorDeconvolution,
    histomicstk.preprocessing.color_deconvolution.ColorConvolution
    """

    im_rgb = 256.0 * np.exp(-im_od * np.log(256.0) / 255.0) - 1.0

    return im_rgb
