import numpy as np


def OpticalDensityInv(I):
    """Transforms input RGB image `I` into optical density space for
    color deconvolution.

    Parameters
    ----------
    I : array_like
        A floating-point image of optical density values obtained
        from OpticalDensityFwd.

    Returns
    -------
    IOut : array_like
        A floating-point multi-channel image with intensity
        values in the range [0, 255].

    See Also
    --------
    histomicstk.preprocessing.color_conversion.OpticalDensityFwd,
    histomicstk.preprocessing.color_deconvolution.ColorDeconvolution,
    histomicstk.preprocessing.color_deconvolution.ColorConvolution
    """

    IOut = np.exp(-(I - 255) * np.log(255) / 255)

    return IOut
