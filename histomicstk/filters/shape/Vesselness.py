import numpy as np

import histomicstk.utils as htk_utils


def vesselness(I, Sigma):
    """
    Calculates vesselness measure for grayscale image `I` at scale `Sigma`.
    Also returns eigenvalues and vectors used for vessel salience filters.

    Parameters
    ----------
    I : array_like
        M x N grayscale image.
    Sigma : double
        standard deviation of gaussian kernel.

    Returns
    -------
    Deviation : array_like
        M x N image of deviation from blob
    Frobenius : array_like
        M x N image of frobenius norm of Hessian - measures presence of
        structure.
    E : array_like
        M x N x 2 eigenvalue image - see eigenvalues.py.
    Theta : array_like
        M x N eigenvector angle image for E(:,:,0) in radians
        see eigenvalues.py. Oriented parallel to vessel structures.

    References
    ----------
    .. [1] Frangi, Alejandro F., et al. "Multiscale vessel enhancement
           filtering." Medical Image Computing and Computer-Assisted
           Interventation. MICCAI98. Springer Berlin Heidelberg,1998.
           130-137.
    """

    # calculate hessian matrix
    H = Sigma**2 * htk_utils.hessian(I, Sigma)

    # calculate eigenvalue image
    E, V1, V2 = htk_utils.eigenvalues(H)

    # compute blobness measures
    Deviation = E[:, :, 0]/(E[:, :, 1] + np.spacing(1))
    Frobenius = np.sqrt(E[:, :, 0]**2 + E[:, :, 1]**2)

    # calculate angles for 'Theta'
    Theta = np.arctan2(V1[:, :, 1], V1[:, :, 0])

    return Deviation, Frobenius, E, Theta
