import numpy as np

from .rgb_to_lab import _lms2lab, _rgb2lms

# Define conversion matrices
_lms2rgb = np.linalg.inv(_rgb2lms)
_lab2lms = np.linalg.inv(_lms2lab)


def lab_to_rgb(im_lab):
    """Transforms an image from LAB to RGB color space

    Parameters
    ----------
    im_lab : array_like
        An image in LAB color space

    Returns
    -------
    im_rgb : array_like
        The RGB representation of the input image 'im_lab'.

    See Also
    --------
    histomicstk.preprocessing.color_conversion.rgb_to_lab,
    histomicstk.preprocessing.color_normalization.reinhard

    References
    ----------
    .. [#] D. Ruderman, T. Cronin, and C. Chiao, "Statistics of cone
       responses to natural images: implications for visual coding,"
       J. Opt. Soc. Am. A 15, 2036-2045 (1998).

    """

    # get input image dimensions
    m = im_lab.shape[0]
    n = im_lab.shape[1]

    # calculate im_lms values from LAB
    im_lab = np.reshape(im_lab, (m * n, 3))
    im_lms = np.dot(_lab2lms, np.transpose(im_lab))

    # calculate RGB values from im_lms
    im_lms = np.exp(im_lms)
    im_lms[im_lms == np.spacing(1)] = 0

    im_rgb = np.dot(_lms2rgb, im_lms)

    # reshape to 3-channel image
    im_rgb = np.reshape(im_rgb.transpose(), (m, n, 3))

    return im_rgb
