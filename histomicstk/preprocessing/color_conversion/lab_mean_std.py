"""Compute statistics for reinhard normalization."""

import numpy as np

from .rgb_to_lab import rgb_to_lab


def lab_mean_std(im_input, mask_out=None):
    """Compute the mean and standard deviation of the intensities.

    ... of each channel of the given RGB image in LAB color space.
    The outputs of this function is for reinhard normalization.

    Parameters
    ----------
    im_input : array_like
        An RGB image

    mask_out : array_like
        if not None, uses numpy masked array functionality to only keep
        non-masked areas when calculating mean and standard deviation.

    Returns
    -------
    mean_lab : array_like
        A 3-element array containing the mean of each channel of the input RGB
        in LAB color space.

    std_lab : array_like
        A 3-element array containing the standard deviation of each channel
        of the input RGB in LAB color space.

    mask_out : array_like, default is None
        if not None, should be (m, n) boolean numpy array.
        This method uses numpy masked array functionality to only use
        non-masked areas in calculations. This is relevant because elements
        like blood, sharpie marker, white space, etc would throw off the
        reinhard normalization by affecting the mean and stdev. Ideally, you
        want to exclude these elements from both the target image (from which
        you calculate target_mu and target_sigma) and from the source image
        to be normalized.

    See Also
    --------
    histomicstk.preprocessing.color_conversion.rgb_to_lab,
    histomicstk.preprocessing.color_conversion.reinhard

    References
    ----------
    .. [#] E. Reinhard, M. Adhikhmin, B. Gooch, P. Shirley, "Color transfer
       between images," in IEEE Computer Graphics and Applications, vol.21,
       no.5,pp.34-41, 2001.
    .. [#] D. Ruderman, T. Cronin, and C. Chiao, "Statistics of cone
       responses to natural images: implications for visual coding,"
       J. Opt. Soc. Am. A vol.15, pp.2036-2045, 1998.

    """
    im_lab = rgb_to_lab(im_input)

    # mask out irrelevant tissue / whitespace / etc
    if mask_out is not None:
        mask_out = mask_out[..., None]
        im_lab = np.ma.masked_array(
            im_lab, mask=np.tile(mask_out, (1, 1, 3)))

    mean_lab = np.array([im_lab[..., i].mean() for i in range(3)])
    std_lab = np.array([im_lab[..., i].std() for i in range(3)])

    return mean_lab, std_lab
