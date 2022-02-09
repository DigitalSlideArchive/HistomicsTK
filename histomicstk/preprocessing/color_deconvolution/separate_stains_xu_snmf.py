"""Placeholder."""
import numpy
import numpy.linalg as np_linalg

import histomicstk.utils as utils

from . import _linalg as htk_linalg


def separate_stains_xu_snmf(im_sda, w_init=None, beta=0.2):
    """Compute the stain matrix for color deconvolution with SNMF.

    ... (sparse non-negative matrix factorization).

    Parameters
    ----------
    im_sda : array_like
        Image (MxNx3) or matrix (3xN) in SDA space for which to compute the
        stain matrix.
    w_init : array_like, default is None
        Initial value for the stain matrix.  if not provided, default
        initialization is used.
    beta : float
        Regularization factor for the sparsity of the deconvolved pixels

    Returns
    -------
    w : array_like
        A 3x3 matrix of stain column vectors

    Note
    ----
    All input pixels are used in the factorization.

    See Also
    --------
    histomicstk.preprocessing.color_deconvolution.color_deconvolution
    histomicstk.preprocessing.color_deconvolution.separate_stains_macenko_pca

    References
    ----------
    .. [#] Van Eycke, Y. R., Allard, J., Salmon, I., Debeir, O., &
           Decaestecker, C. (2017).  Image processing in digital pathology: an
           opportunity to solve inter-batch variability of immunohistochemical
           staining.  Scientific Reports, 7.
    .. [#] Xu, J., Xiang, L., Wang, G., Ganesan, S., Feldman, M., Shih, N. N.,
           ... & Madabhushi, A. (2015). Sparse Non-negative Matrix Factorization
           (SNMF) based color unmixing for breast histopathological image
           analysis.  Computerized Medical Imaging and Graphics, 46, 20-29.

    """
    import nimfa

    # Image matrix
    m = utils.convert_image_to_matrix(im_sda)
    m = utils.exclude_nonfinite(m)
    factorization = \
        nimfa.Snmf(m, rank=m.shape[0] if w_init is None else w_init.shape[1],
                   W=w_init,
                   H=None if w_init is None else np_linalg.pinv(w_init).dot(m),
                   beta=beta)
    factorization.factorize()
    return htk_linalg.normalize(numpy.array(factorization.W))
