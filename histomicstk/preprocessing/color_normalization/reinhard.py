"""Placeholder."""
from histomicstk.preprocessing import color_conversion
import numpy as np


def reinhard(
        im_src, target_mu, target_sigma, src_mu=None, src_sigma=None,
        mask_out=None):
    """Perform Reinhard color normalization.

    Transform the color characteristics of an image to a desired standard.
    The standard is defined by the mean and standard deviations of the target
    image in LAB color space defined by Ruderman. The input image is converted
    to Ruderman's LAB space, the LAB channels are each centered and scaled to
    zero-mean unit variance, and then rescaled and shifted to match the target
    image statistics. If the LAB statistics for the input image are provided
    (`src_mu` and `src_sigma`) then these will be used for normalization,
    otherwise they will be derived from the input image `im_src`.

    Parameters
    ----------
    im_src : array_like
        An RGB image

    target_mu : array_like
        A 3-element array containing the means of the target image channels
        in LAB color space.

    target_sigma : array_like
        A 3-element array containing the standard deviations of the target
        image channels in LAB color space.

    src_mu : array_like, optional
        A 3-element array containing the means of the source image channels in
        LAB color space. Used with reinhard_stats for uniform normalization of
        tiles from a slide.

    src_sigma : array, optional
        A 3-element array containing the standard deviations of the source
        image channels in LAB color space. Used with reinhard_stats for
        uniform normalization of tiles tiles from a slide.

    mask_out : array_like, default is None
        if not None, should be (m, n) boolean numpy array.
        This method uses numpy masked array functionality to only use
        non-masked areas in calculations. This is relevant because elements
        like blood, sharpie marker, white space, etc would throw off the
        reinhard normalization by affecting the mean and stdev. Ideally, you
        want to exclude these elements from both the target image (from which
        you calculate target_mu and target_sigma) and from the source image
        to be normalized.

    Returns
    -------
    im_normalized : array_like
        Color Normalized RGB image

    See Also
    --------
    histomicstk.preprocessing.color_conversion.rgb_to_lab,
    histomicstk.preprocessing.color_conversion.lab_to_rgb

    References
    ----------
    .. [#] E. Reinhard, M. Adhikhmin, B. Gooch, P. Shirley, "Color transfer
       between images," in IEEE Computer Graphics and Applications, vol.21,
       no.5,pp.34-41, 2001.
    .. [#] D. Ruderman, T. Cronin, and C. Chiao, "Statistics of cone responses
       to natural images: implications for visual coding," J. Opt. Soc. Am. A
       vol.15, pp.2036-2045, 1998.

    """
    # convert input image to LAB color space
    im_lab = color_conversion.rgb_to_lab(im_src)

    # mask out irrelevant tissue / whitespace / etc
    if mask_out is not None:
        mask_out = mask_out[..., None]
        im_lab = np.ma.masked_array(
            im_lab, mask=np.tile(mask_out, (1, 1, 3)))

    # calculate src_mu and src_sigma if either is not provided
    if (src_mu is None) or (src_sigma is None):
        src_mu = [im_lab[..., i].mean() for i in range(3)]
        src_sigma = [im_lab[..., i].std() for i in range(3)]

    # scale to unit variance
    for i in range(3):
        im_lab[:, :, i] = (im_lab[:, :, i] - src_mu[i]) / src_sigma[i]

    # rescale and recenter to match target statistics
    for i in range(3):
        im_lab[:, :, i] = im_lab[:, :, i] * target_sigma[i] + target_mu[i]

    # convert back to RGB colorspace
    im_normalized = color_conversion.lab_to_rgb(im_lab)
    im_normalized[im_normalized > 255] = 255
    im_normalized[im_normalized < 0] = 0

    # return masked values and reconstruct unmasked LAB image
    if mask_out is not None:
        im_normalized = im_normalized.data
        for i in range(3):
            original = im_src[:, :, i].copy()
            new = im_normalized[:, :, i].copy()
            original[np.not_equal(mask_out[:, :, 0], True)] = 0
            new[mask_out[:, :, 0]] = 0
            im_normalized[:, :, i] = new + original

    im_normalized = im_normalized.astype(np.uint8)

    return im_normalized
