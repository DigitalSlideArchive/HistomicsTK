import numpy as np


def OpticalDensityFwd(I):
    """Transforms input RGB image "I" into optical density space
    for color deconvolution.

    Parameters
    ----------
    I : array_like
        A floating-point RGB image with intensity ranges of [0, 255].

    Returns
    -------
    IOut : array_like
        A floating-point image of corresponding optical density values.


    See Also
    --------
    OpticalDensityInv,
    histomicstk.preprocessing.color_deconvolution.ColorDeconvolution,
    histomicstk.preprocessing.color_deconvolution.ColorConvolution
    """

    IOut = -(255 * np.log(I / 255)) / np.log(255)

    return IOut
