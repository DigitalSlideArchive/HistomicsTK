import numpy as np


def RudermanLABFwd(I):
    """Transforms an image from RGB to LAB color space

    Parameters
    ----------
    I : array_like
        An RGB image

    Returns
    -------
    I_LAB : array_like
        LAB representation of the input image `I`.

    See Also
    --------
    histomicstk.preprocessing.color_conversion.RudermanLABInv,
    histomicstk.preprocessing.color_normalization.ReinhardNorm

    References
    ----------
    .. [1] D. Ruderman, T. Cronin, and C. Chiao, "Statistics of cone responses
       to natural images: implications for visual coding," J. Opt. Soc. Am. A
       vol.15, pp.2036-2045, 1998.
    """

    # get input image dimensions
    m = I.shape[0]
    n = I.shape[1]

    # define conversion matrices
    RGB2LMS = np.array([[0.3811, 0.5783, 0.0402],
                        [0.1967, 0.7244, 0.0782],
                        [0.0241, 0.1288, 0.8444]])
    LMS2LAB = np.array([[1 / (3**0.5), 0, 0],
                        [0, 1 / (6**0.5), 0],
                        [0, 0, 1 / (2**0.5)]]).dot(np.array([[1, 1, 1],
                                                            [1, 1, -2],
                                                            [1, -1, 0]]))

    # calculate LMS values from RGB
    I = np.reshape(I, (m * n, 3))
    LMS = np.dot(RGB2LMS, np.transpose(I))
    LMS[LMS == 0] = np.spacing(1)
    logLMS = np.log(LMS)

    # calculate LAB values from LMS
    I_LAB = LMS2LAB.dot(logLMS)

    # reshape to 3-channel image
    I_LAB = np.reshape(I_LAB.transpose(), (m, n, 3))

    return I_LAB
