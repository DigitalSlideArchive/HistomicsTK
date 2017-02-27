import numpy as np

from .membraneness import membraneness


def membranefilter(im_input, sigmas, beta, c):
    """
    Implements a membrane-detecting filter using eigenvalue analysis of
    scale-space blur image Hessians.

    Parameters
    ----------
    im_input : array_like
        M x N grayscale image. Objects to be detected are dark.
    sigmas : array_like
        vector of gaussian deviations (scales) at which response is to be
        evaluated.
    beta : double
        Scaling property of deviation from blob. Decrease to increase
        importance of deviation in final score. Default value = 1/2.
    c : double
        Scaling property of quadratic curvature magnitude. Decrease to
        increase the importance of curvature magnitude in final score.
        Default value = 5.

    Returns
    -------
    Response : array_like
        M x N intensity image of membrane filter response.
    Thetas : array_like
        M x N image of membrane parallel angles in radians.

    References
    ----------
    .. [1] Frangi, Alejandro F., et al. "Multiscale vessel enhancement
           filtering." Medical Image Computing and Computer-Assisted
           Interventation. MICCAI98. Springer Berlin Heidelberg,1998.
           130-137.
    """

    # get size of input image 'I'
    M = im_input.shape[0]
    N = im_input.shape[1]

    # convert image to double if necessary
    if np.issubdtype(im_input.dtype, np.uint8):
        im_input = im_input.astype('double')
    else:
        im_input = im_input/im_input.max()

    # initialize filter response, angles
    Response = np.zeros((M, N, len(sigmas)))
    Thetas = np.zeros((M, N, len(sigmas)))

    # calculate blob score at multiple scales
    for i in range(0, len(sigmas)):
        # blob score
        Deviation, Frobenius, E, Thetas[:, :, i] = membraneness(
            im_input, sigmas[i]
        )
        # combine deviation and curvature into single score
        R = np.exp(-0.5*(Deviation/beta)**2) * \
            (np.ones((M, N))-np.exp(-0.5*(Frobenius/c)**2))
        # clear response where lambda_2 > 0
        R[np.where(E[:, :, 1] < 0)] = 0
        Response[:, :, i] = R

    # calculate final blob response as maximum over all scales
    Response, Index = Response.max(axis=2), Response.argmax(axis=2)

    sizeRx = Response.shape[0]
    sizeRy = Response.shape[1]

    mats = np.arange(0, sizeRx*sizeRy)
    thetaIndex = np.unravel_index(
        mats + np.ravel(Index)*M*N, (sizeRx, sizeRy, 3)
    )

    Thetas = np.reshape(Thetas[thetaIndex], (sizeRx, sizeRy))

    return Response, Thetas
