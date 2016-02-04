import collections
import numpy as np
import sklearn.cluster as cl
import scipy.signal as signal


def GaussianVoting(I, rmax=35, rmin=10, sSigma=5, Tau=5, bw=15, Psi=0.3):
    """Performs nuclear detection using Gaussian kernel voting.

    Uses a gaussian kernel to localize the centroids of cell nuclei. Takes
    as input a hematoxylin-deconvolved image and uses the gradient signal to
    cast directed votes towards the center of cell nuclei. These votes are
    blurred by a gaussian kernel, and are spatially clustered using the
    mean-shift algorithm. Convolutions are performed separably to reduce
    compute time.

    Parameters
    ----------
    I : array_like
        A hematoxylin intensity image obtained from ColorDeconvolution.
    rmax : double
        Upper-limit on voting area extent. Default value = 35.
    rmin : double
        Lower-limit on voting area extent. Default value = 10.
    sSigma : double
        Standard deviation of smoothing kernel used in gradient calculation.
        Default value = 5.
    Tau : double
        Lower limit on gradient magnitude for casting a vote. Default
        value = 5.
    bw : double
        Bandwidth parameter for mean-shift clustering. Default value = 15.
    Psi : double
        Lower limit threshold on votes. Expressed as a percentage of
        the maximum vote, ranges from [0, 1). Default value = 0.3.

    Returns
    -------
    Centroids : array_like
        An N x 2 array defining the (x,y) coordinates of cell centroids.
    Votes : array_like
        An intensity image containing the blurred votes obtained by voting.

    Notes
    -----
    Return values are returned as a namedtuple

    See Also
    --------
    ColorDeconvolution

    References
    ----------
    .. [1] X. Qi, F. Xing, D.J. Foran, L. Yang, "Robust Segmentation of
           Overlapping Cells in Histopathology Specimens Using Parallel
           Seed Detection and Repulsive Level Set," in IEEE Transactions
           on Biomedical Engineering, vol.59,no.23,pp.754-65, 2011.
    """

    # calculate standard deviation of voting kernel
    vSigma = (rmax - rmin) / 3

    # calculate voting radius
    r = (rmax + rmin) / 2

    # generate separable gaussian derivative kernels
    x = np.linspace(0, 2 * 3 * sSigma, 2 * 3 * sSigma + 1)
    y = np.linspace(0, 2 * 3 * sSigma, 2 * 3 * sSigma + 1)
    x -= 2 * 3 * sSigma / 2  # center independent variables at zero
    y -= 2 * 3 * sSigma / 2
    x = np.reshape(x, (1, x.size))  # reshape to 2D row and column vectors
    y = np.reshape(y, (y.size, 1))
    xGx = 2 * x / (sSigma**2) * np.exp(-x**2 / (2 * sSigma**2)) \
        / (sSigma * (2 * np.pi) ** 0.5)
    yGx = np.exp(-y**2 / (2 * sSigma**2)) / ((2 * np.pi) ** 0.5 * sSigma)
    xGy = np.exp(-x**2 / (2 * sSigma**2)) / ((2 * np.pi) ** 0.5 * sSigma)
    yGy = 2 * y / (sSigma**2) * np.exp(-y**2 / (2 * sSigma**2)) \
        / (sSigma * (2 * np.pi) ** 0.5)

    # smoothed gradients of input image
    dX = signal.convolve2d(I, xGx, mode='same', boundary='symm')
    dX = signal.convolve2d(dX, yGx, mode='same', boundary='symm')
    dY = signal.convolve2d(I, xGy, mode='same', boundary='symm')
    dY = signal.convolve2d(dY, yGy, mode='same', boundary='symm')
    dMag = (dX**2 + dY**2)**0.5

    # threshold gradient image to identify voting pixels
    dMask = dMag >= Tau
    Voting = dMask.nonzero()

    # initialize voting field - pad by 'r' along each edge
    V = np.zeros((I.shape[0] + 2 * r, I.shape[1] + 2 * r))
    for i in range(Voting[0].size):

        # calculate center point of voting region
        mux = round(Voting[1][i] + r * dX[Voting[0][i]][Voting[1][i]] /
                    dMag[Voting[0][i]][Voting[1][i]])
        muy = round(Voting[0][i] + r * dY[Voting[0][i]][Voting[1][i]] /
                    dMag[Voting[0][i]][Voting[1][i]])

        # enter weighted votes at these locations
        V[r+muy, r+mux] = V[r+muy, r+mux] + dMag[Voting[0][i]][Voting[1][i]]

    # create voting kernel
    x = np.linspace(0, 2*3*vSigma, 2*3*vSigma + 1)  # independent variables
    y = np.linspace(0, 2*3*vSigma, 2*3*vSigma + 1)
    x -= 2*3*vSigma/2  # center at zero
    y -= 2*3*vSigma/2
    x = np.reshape(x, (1, x.size))  # reshape to 2D row and column vectors
    y = np.reshape(y, (y.size, 1))
    xK = np.exp(-x**2 / (2 * vSigma**2)) / (sSigma * (2 * np.pi) ** 0.5)
    yK = np.exp(-y**2 / (2 * vSigma**2)) / (sSigma * (2 * np.pi) ** 0.5)

    # perform convolutions with voting kernel for vote-smoothing
    V = signal.convolve2d(V, xK, mode='full', boundary='fill')
    V = signal.convolve2d(V, yK, mode='full', boundary='fill')

    # crop voting image to size of original input image
    V = V[r+np.floor(x.size/2):-(r+np.ceil(x.size/2)),
          r+np.floor(x.size/2):-(r+np.ceil(x.size/2))]

    # generate sets of potential seed points
    Seeds = [None]*np.arange(np.floor(10 * Psi) / 10, 0.9, 0.1).size
    for i, p in enumerate(np.arange(np.floor(10 * Psi) / 10, 0.9, 0.1)):
        Points = np.nonzero(V >= p * V.max())
        Seeds[i] = np.column_stack((Points[0].transpose(),
                                    Points[1].transpose()))

    # concatenate seed point lists
    Seeds = np.vstack(Seeds)

    # run mean-shift algorithm to collect
    ms = cl.MeanShift(bandwidth=bw, bin_seeding=True)
    ms.fit(Seeds)

    # build output tuple
    Output = collections.namedtuple('Output', ['Centroids', 'Votes'])
    Nuclei = Output(ms.cluster_centers_, V)

    return Nuclei
