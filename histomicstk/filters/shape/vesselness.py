import numpy as np

import histomicstk.utils as htk_utils


def vesselness(im_input, sigma):
    """
    Calculates vesselness measure for grayscale image `im_input` at scale `sigma`.
    Also returns eigenvalues and vectors used for vessel salience filters.

    Parameters
    ----------
    im_input : array_like
        M x N grayscale image.
    sigma : double
        standard deviation of gaussian kernel.

    Returns
    -------
    Deviation : array_like
        M x N image of deviation from blob
    Frobenius : array_like
        M x N image of frobenius norm of Hessian - measures presence of
        structure.
    E : array_like
        M x N x 2 eigenvalue image - see eigen.py.
    Theta : array_like
        M x N eigenvector angle image for E(:,:,0) in radians
        see eigen.py. Oriented parallel to vessel structures.

    References
    ----------
    .. [#] Frangi, Alejandro F., et al. "Multiscale vessel enhancement
       filtering." Medical Image Computing and Computer-Assisted
       Interventation. MICCAI98. Springer Berlin Heidelberg,1998. 130-137.

    """
    # calculate hessian matrix
    H = sigma ** 2 * htk_utils.hessian(im_input, sigma)

    # calculate eigenvalue image
    E, V1, V2 = htk_utils.eigen(H)

    # compute blobness measures
    Deviation = E[:, :, 0] / (E[:, :, 1] + np.spacing(1))
    Frobenius = np.sqrt(E[:, :, 0]**2 + E[:, :, 1]**2)

    # calculate angles for 'Theta'
    Theta = np.arctan2(V1[:, :, 1], V1[:, :, 0])

    return Deviation, Frobenius, E, Theta
