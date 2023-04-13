import numpy as np


def find_stain_index(reference, w):
    """Find the index of the stain column vector in w corresponding to the
    reference vector.  Useful in connection with adaptive
    deconvolution routines in order to find the column corresponding
    with a certain expected stain.

    Parameters
    ----------
    reference : array_like
        1D array that is the stain vector to find
    w : array_like
        2D array of columns the same size as reference.
        The columns should be normalized.

    Returns
    -------
    i : int
        Column of w corresponding to reference

    Notes
    -----
    The index of the vector with the smallest distance is returned.

    See Also
    --------
    histomicstk.preprocessing.color_deconvolution.separate_stains_macenko_pca
    histomicstk.preprocessing.color_deconvolution.color_deconvolution

    """
    dists = [np.linalg.norm(w[i] - reference) for i in range(w.shape[0])]
    return np.argmin(dists)
