import numpy as np


def RudermanLABInv(I):
    """Transforms an image from LAB to RGB color space

    Parameters
    ----------
    I : array_like
        An image in LAB color space

    Returns
    -------
    I_RGB : array_like
        The RGB representation of the input image 'I'.

    See Also
    --------
    RudermanLABFwd, ReinhardNorm
    
    References
    ----------
    .. [1] D. Ruderman, T. Cronin, and C. Chiao, "Statistics of cone responses
       to natural images: implications for visual coding," J. Opt. Soc. Am. A
       15, 2036-2045 (1998).
    """

    # get input image dimensions
    m = I.shape[0]
    n = I.shape[1]
    
    # define conversion matrices
    LAB2LMS = np.array([[1, 1, 1],
                        [1, 1, -1],
                        [1, -2, 0]]).dot(np.array([[1/(3**(0.5)), 0, 0],
                                                   [0, 1/(6**(0.5)), 0],
                                                   [0, 0, 1/(2**(0.5))]]))
    LMS2RGB = np.array([[4.4679, -3.5873, 0.1193],
                        [-1.2186, 2.3809, -0.1624],
                        [0.0497, -0.2439, 1.2045]])
    
    # calculate LMS values from LAB
    I = np.reshape(I, (m*n, 3))
    LMS = np.dot(LAB2LMS, np.transpose(I))
    expLMS = np.exp(LMS)

    # calculate RGB values from LMS
    I_RGB = LMS2RGB.dot(expLMS)
    
    #reshape to 3-channel image
    I_RGB = np.reshape(I_RGB.transpose(), (m, n, 3))
    
    return(I_RGB)
