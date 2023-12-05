import numpy as np

from . import _linalg as linalg


def find_stain_index(reference, w):
    """Identify the stain vector in w that best aligns with the reference vector.

    This is used with adaptive deconvolution routines where the order of returned stain
    vectors is not guaranteed. This function identifies the stain vector of w that most
    closely aligns with the provided reference.

    Parameters
    ----------
    reference : array_like
        1D array representing the stain vector query.
    w : array_like
        3xN array of where columns represent stain vectors to search.

    Returns
    -------
    i : int
        Column index of stain vector with best alignment to reference.

    Notes
    -----
    Vectors are normalized to unit-norm prior to comparison using dot product. Alignment
    is determined by vector angles and not distances.

    See Also
    --------
    histomicstk.preprocessing.color_deconvolution.separate_stains_macenko_pca
    histomicstk.preprocessing.color_deconvolution.color_deconvolution

    """
    dot_products = np.dot(
        linalg.normalize(np.array(reference)), linalg.normalize(np.array(w)),
    )
    return np.argmax(np.abs(dot_products))
