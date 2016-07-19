import numpy as np

from Membraneness import Membraneness


def MembraneFilter(I, Sigmas, Beta=0.5, C=5):
    """
    Implements a membrane-detecting filter using eigenvalue analysis of
    scale-space blur image Hessians.

    Parameters
    ----------
    I : array_like
        M x N grayscale image. Objects to be detected are dark.
    Sigmas : array_like
        vector of gaussian deviations (scales) at which response is to be
        evaluated.
    Beta : double
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
    """

    # get size of input image 'I'
    M = I.shape[0]
    N = I.shape[1]

    # convert image to double if necessary
    if np.issubdtype(I.dtype, np.uint8):
        I = I.astype('double')
    else:
        I = I/I.max()

    # initialize filter response, angles
    Response = np.zeros((M, N, len(Sigmas)))
    Thetas = np.zeros((M, N, len(Sigmas)))

    # calculate blob score at multiple scales
    for i in range(0, len(Sigmas)):
        # blob score
        Deviation, Frobenius, E, Thetas[:, :, i] = Membraneness(I, Sigmas[i])
        # combine deviation and curvature into single score
        R = np.exp(-0.5*(Deviation/Beta)**2) * \
            (np.ones((M, N))-np.exp(-0.5*(Frobenius/C)**2))
        # clear response where lambda_2 > 0
        R[np.where(E[:, :, 1] < 0)] = 0
        Response[:, :, i] = R

    # calculate final blob response as maximum over all scales
    Response, Index = Response.max(axis=2), Response.argmax(axis=2)

    sizeofRx = Response.shape[0]
    sizeofRy = Response.shape[1]

    mats = np.arange(0, sizeofRx*sizeofRy)
    thetaIndex = np.unravel_index(
        mats + np.ravel(Index)*M*N, (sizeofRx, sizeofRy)
    )
    Thetas = np.reshape(Thetas[thetaIndex], (sizeofRx, sizeofRy))

    return Response, Thetas
