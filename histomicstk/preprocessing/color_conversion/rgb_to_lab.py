import numpy as np


# define conversion matrices
_rgb2lms = np.array([[0.3811, 0.5783, 0.0402],
                     [0.1967, 0.7244, 0.0782],
                     [0.0241, 0.1288, 0.8444]])

_lms2lab = np.dot(
    np.array([[1 / (3**0.5), 0, 0],
              [0, 1 / (6**0.5), 0],
              [0, 0, 1 / (2**0.5)]]),
    np.array([[1, 1, 1],
              [1, 1, -2],
              [1, -1, 0]])
)


def rgb_to_lab(im_rgb):
    """Transforms an image from RGB to LAB color space

    Parameters
    ----------
    im_rgb : array_like
        An RGB image

    Returns
    -------
    im_lab : array_like
        LAB representation of the input image `im_rgb`.

    See Also
    --------
    histomicstk.preprocessing.color_conversion.lab_to_rgb,
    histomicstk.preprocessing.color_normalization.reinhard

    References
    ----------
    .. [#] D. Ruderman, T. Cronin, and C. Chiao, "Statistics of cone
       responses to natural images: implications for visual coding,"
       J. Opt. Soc. Am. A vol.15, pp.2036-2045, 1998.

    """

    # get input image dimensions
    m = im_rgb.shape[0]
    n = im_rgb.shape[1]

    # calculate im_lms values from RGB
    im_rgb = np.reshape(im_rgb, (m * n, 3))
    im_lms = np.dot(_rgb2lms, np.transpose(im_rgb))
    im_lms[im_lms == 0] = np.spacing(1)

    # calculate LAB values from im_lms
    im_lab = np.dot(_lms2lab, np.log(im_lms))

    # reshape to 3-channel image
    im_lab = np.reshape(im_lab.transpose(), (m, n, 3))

    return im_lab
