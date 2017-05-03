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

    stain0 = w[:, 0]
    stain1 = w[:, 1]
    stain2 = np.cross(stain0, stain1)
    # Normalize new vector to have unit norm
    return np.array([stain0, stain1, stain2 / np.linalg.norm(stain2)]).T
