from .macenko_stain_matrix import macenko_stain_matrix
from ..color_conversion import rgb_to_sda

def rgb_macenko_stain_matrix(im_rgb, I_0, *args, **kwargs):
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
    histomicstk.preprocessing.color_deconvolution.macenko_stain_matrix.
    im_sda is computed and passed as part of this routine.

    See Also
    --------
    histomicstk.preprocessing.color_deconvolution.macenko_stain_matrix

    """
    im_sda = rgb_to_sda(im_rgb, I_0)
    return macenko_stain_matrix(im_sda, *args, **kwargs)
