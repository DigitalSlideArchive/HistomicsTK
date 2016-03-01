import numpy as np


def ComplementStainMatrix(W):
    """Generates a complemented stain matrix
    Used to fill out empty columns of a stain matrix for use with
    ColorDeconvolution. Replaces right-most column with normalized
    cross-product of first two columns.

    Parameters
    ----------
    W : array_like
        A 3x3 stain calibration matrix with stain color vectors in columns.

    Returns
    -------
    WComp : array_like
        A 3x3 complemented stain calibration matrix with a third
        orthogonal column.

    See Also
    --------
    ColorDeconvolution
    """

    # copy input to output for initialization
    WComp = W

    # calculate directed cross-product of first two columns
    if (W[0, 0]**2 + W[0, 1]**2) > 1:
        WComp[0, 2] = 0
    else:
        WComp[0, 2] = (1 - (W[0, 0]**2 + W[0, 1]**2))**0.5

    if (W[1, 0]**2 + W[1, 1]**2) > 1:
        WComp[1, 2] = 0
    else:
        WComp[1, 2] = (1 - (W[1, 0]**2 + W[1, 1]**2))**0.5

    if (W[2, 0]**2 + W[2, 1]**2) > 1:
        WComp[2, 2] = 0
    else:
        WComp[2, 2] = (1 - (W[2, 0]**2 + W[2, 1]**2))**0.5

    # normalize new vector to unit-norm
    WComp[:, 2] = WComp[:, 2] / np.linalg.norm(WComp[:, 2])

    return WComp
