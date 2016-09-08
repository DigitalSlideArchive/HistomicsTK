import numpy as np


def Del2(X):
    """Discrete Laplacian with edge-value extrapolation.

    Calculates the discrete Laplacian of an input image. Edge values are
    calculated by using linear extrapolation of second differences. This is
    consistent with the way that Matlab calculates the discrete Laplacian.

    Parameters
    ----------
    X : array_like
        A floating-point intensity image.

    Returns
    -------
    L : array_like
        The discrete Laplacian of L.

    See Also
    --------
    histomicstk.segmentation.level_set.DregEdge,
    histomicstk.segmentation.level_set.ChanVese
    """

    # initialize output
    L = np.zeros(X.shape)

    # process rows first
    D = np.diff(X, axis=0)
    S = np.zeros(X.shape)
    S[1:-1, :] = (D[1:, :] - D[0:-1, :])/2
    if X.shape[0] > 3:
        S[0, :] = 2 * S[1, :] - S[2, :]
        S[-1, :] = 2 * S[-2, :] - S[-3, :]
    elif X.shape[0] == 3:
        S[0, :] = S[1, :]
        S[-1, :] = S[1, :]
    else:
        S[0, :] = 0
        S[-1, :] = 0
    L += S

    # process columns
    D = np.diff(X, axis=1)
    S = np.zeros(X.shape)
    S[:, 1:-1] = (D[:, 1:] - D[:, 0:-1])/2
    if X.shape[1] > 3:
        S[:, 0] = 2 * S[:, 1] - S[:, 2]
        S[:, -1] = 2 * S[:, -2] - S[:, -3]
    elif X.shape[1] == 3:
        S[0, :] = S[:, 1]
        S[:, -1] = S[:, 1]
    else:
        S[:, 0] = 0
        S[:, -1] = 0
    L += S

    return L / 2
