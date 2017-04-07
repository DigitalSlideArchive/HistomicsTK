import numpy as np


def complement_stain_matrix(w):
    """Generates a complemented stain matrix
    Used to fill out empty columns of a stain matrix for use with
    color_deconvolution. Replaces right-most column with normalized
    cross-product of first two columns.

    Parameters
    ----------
    w : array_like
        A 3x3 stain calibration matrix with stain color vectors in columns.

    Returns
    -------
    w_comp : array_like
        A 3x3 complemented stain calibration matrix with a third
        orthogonal column.

    See Also
    --------
    histomicstk.preprocessing.color_deconvolution.color_deconvolution
    """

    # copy input to output for initialization
    w_comp = w.copy()

    # calculate directed cross-product of first two columns
    if (w[0, 0]**2 + w[0, 1]**2) > 1:
        w_comp[0, 2] = 0
    else:
        w_comp[0, 2] = (1 - (w[0, 0] ** 2 + w[0, 1] ** 2)) ** 0.5

    if (w[1, 0]**2 + w[1, 1]**2) > 1:
        w_comp[1, 2] = 0
    else:
        w_comp[1, 2] = (1 - (w[1, 0] ** 2 + w[1, 1] ** 2)) ** 0.5

    if (w[2, 0]**2 + w[2, 1]**2) > 1:
        w_comp[2, 2] = 0
    else:
        w_comp[2, 2] = (1 - (w[2, 0] ** 2 + w[2, 1] ** 2)) ** 0.5

    # normalize new vector to unit-norm
    w_comp[:, 2] = w_comp[:, 2] / np.linalg.norm(w_comp[:, 2])

    return w_comp
