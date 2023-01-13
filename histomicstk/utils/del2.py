import numpy as np


def del2(im_input):
    """Discrete Laplacian with edge-value extrapolation.

    Calculates the discrete Laplacian of an input image. Edge values are
    calculated by using linear extrapolation of second differences. This is
    consistent with the way that Matlab calculates the discrete Laplacian.

    Parameters
    ----------
    im_input : array_like
        A floating-point intensity image.

    Returns
    -------
    im_lap : array_like
        The discrete Laplacian of im_lap.

    See Also
    --------
    histomicstk.segmentation.level_set.reg_edge

    """

    # initialize output
    im_lap = np.zeros(im_input.shape)

    # process rows first
    D = np.diff(im_input, axis=0)
    S = np.zeros(im_input.shape)
    S[1:-1, :] = (D[1:, :] - D[0:-1, :]) / 2
    if im_input.shape[0] > 3:
        S[0, :] = 2 * S[1, :] - S[2, :]
        S[-1, :] = 2 * S[-2, :] - S[-3, :]
    elif im_input.shape[0] == 3:
        S[0, :] = S[1, :]
        S[-1, :] = S[1, :]
    else:
        S[0, :] = 0
        S[-1, :] = 0
    im_lap += S

    # process columns
    D = np.diff(im_input, axis=1)
    S = np.zeros(im_input.shape)
    S[:, 1:-1] = (D[:, 1:] - D[:, 0:-1]) / 2
    if im_input.shape[1] > 3:
        S[:, 0] = 2 * S[:, 1] - S[:, 2]
        S[:, -1] = 2 * S[:, -2] - S[:, -3]
    elif im_input.shape[1] == 3:
        S[0, :] = S[:, 1]
        S[:, -1] = S[:, 1]
    else:
        S[:, 0] = 0
        S[:, -1] = 0
    im_lap += S

    return im_lap / 2
