from .separate_stains_xu_snmf import separate_stains_xu_snmf
from ..color_conversion import rgb_to_sda


def rgb_separate_stains_xu_snmf(im_rgb, I_0, *args, **kwargs):
    """Compute the stain matrix for color deconvolution with SNMF from an
    RGB image or matrix.

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
    This is a thin wrapper around separate_stains_xu_snmf and as such
    the w_init parameter must be manually passed.

    For additional input arguments and documentation, please see
    histomicstk.preprocessing.color_deconvolution.separate_stains_xu_snmf.
    im_sda is computed and passed as part of this routine.

    See Also
    --------
    histomicstk.preprocessing.color_deconvolution.separate_stains_xu_snmf

    """
    im_sda = rgb_to_sda(im_rgb, I_0)
    return separate_stains_xu_snmf(im_sda, *args, **kwargs)
