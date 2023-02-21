import numpy as np


def glog(im_input, alpha=1, range=None, theta=np.pi / 4, tau=0.6, eps=0.6):
    """Performs generalized Laplacian of Gaussian blob detection.

    Parameters
    ----------
    im_input : array_like
        A hematoxylin intensity image obtained from ColorDeconvolution.
    alpha : double
        A positive scalar used to normalize the gLoG filter responses. Controls
        the blob-center detection and eccentricities of detected blobs. Larger
        values emphasize more eccentric blobs. Default value = 1.
    range : array_like
        Scale range
    theta : double
        Angular increment for rotating gLoG filters. Default value = np.pi / 6.
    tau : double
        Tolerance for counting pixels in determining optimal scale SigmaC
    eps : double
        range to define SigmaX surrounding SigmaC

    Returns
    -------
    Rsum : array_like
        Sum of filter responses at specified scales and orientations
    Maxima: : array_like
        A binary mask highlighting maxima pixels

    Notes
    -----
    Return values are returned as a namedtuple

    References
    ----------
    .. [#] H. Kong, H.C. Akakin, S.E. Sarma, "A Generalized Laplacian of
       Gaussian Filter for Blob Detection and Its Applications," in IEEE
       Transactions on Cybernetics, vol.43,no.6,pp.1719-33, 2013.

    """
    import scipy.ndimage as ndi
    from skimage import morphology

    range = np.linspace(1.5, 3, int(np.round((3 - 1.5) / 0.2)) + 1) if range is None else range

    # initialize sigma
    Sigma = np.exp(range)

    # generate circular LoG scale-space to determine range of SigmaX
    l_g = 0
    H = []
    Bins = []
    Min = np.zeros((len(Sigma), 1))
    Max = np.zeros((len(Sigma), 1))
    for i, s in enumerate(Sigma):
        Response = s**2 * ndi.gaussian_laplace(im_input, s, output=None,
                                               mode='constant',
                                               cval=0.0)
        Min[i] = Response.min()
        Max[i] = Response.max()
        Bins.append(np.arange(0.01 * np.floor(Min[i] / 0.01),
                    0.01 * np.ceil(Max[i] / 0.01) + 0.01, 0.01))
        Hist = np.histogram(Response, Bins[i])
        H.append(Hist[0])
        if Max[i] > l_g:
            l_g = Max[i]

    # re-normalized based on global max and local min, count threshold pixels
    Zeta = np.zeros((len(Sigma), 1))
    for i, s in enumerate(Sigma):
        Bins[i] = (Bins[i] - Min[i]) / (l_g - Min[i])
        Zeta[i] = np.sum(H[i][Bins[i][0:-1] > tau])

    # identify best scale SigmaC based on maximum circular response
    Index = np.argmax(Zeta)

    # define range for SigmaX
    XRange = range(max(Index - 2, 0), min(len(range), Index + 2) + 1)
    SigmaX = np.exp(range[XRange])

    # define rotation angles
    Thetas = np.linspace(0, np.pi - theta, int(np.round(np.pi / theta)))

    # loop over SigmaX, SigmaY and then angle, summing up filter responses
    Rsum = np.zeros(im_input.shape)
    for i, Sx in enumerate(SigmaX):
        YRange = range(0, XRange[i])
        SigmaY = np.exp(range[YRange])
        for Sy in SigmaY:
            for Th in Thetas:
                Kernel = glogkernel(Sx, Sy, Th)
                Kernel *= (1 + np.log(Sx) ** alpha) * (1 + np.log(Sy) ** alpha)
                Rsum += ndi.convolve(im_input, Kernel, mode='constant', cval=0.0)
                print(Sx, Sy, Th)
        Kernel = glogkernel(Sx, Sx, 0)
        Kernel *= (1 + np.log(Sx) ** alpha) * (1 + np.log(Sx) ** alpha)
        Rsum += ndi.convolve(im_input, Kernel, mode='constant', cval=0.0)
        print(Sx, Sx, 0)

    # detect local maxima
    Disk = morphology.disk(3 * np.exp(range[Index]))
    Maxima = ndi.maximum_filter(Rsum, footprint=Disk)
    Maxima = Rsum == Maxima

    return Rsum, Maxima


def glogkernel(sigma_x, sigma_y, theta):

    N = np.ceil(2 * 3 * sigma_x)
    X, Y = np.meshgrid(np.linspace(0, N, int(N + 1)) - N / 2,
                       np.linspace(0, N, int(N + 1)) - N / 2)
    a = np.cos(theta) ** 2 / (2 * sigma_x ** 2) + \
        np.sin(theta) ** 2 / (2 * sigma_y ** 2)
    b = -np.sin(2 * theta) / (4 * sigma_x ** 2) + \
        np.sin(2 * theta) / (4 * sigma_y ** 2)
    c = np.sin(theta) ** 2 / (2 * sigma_x ** 2) + \
        np.cos(theta) ** 2 / (2 * sigma_y ** 2)
    D2Gxx = ((2 * a * X + 2 * b * Y)**2 - 2 * a) * np.exp(-(a * X**2 + 2 * b * X * Y + c * Y**2))
    D2Gyy = ((2 * b * X + 2 * c * Y)**2 - 2 * c) * np.exp(-(a * X**2 + 2 * b * X * Y + c * Y**2))
    Gaussian = np.exp(-(a * X**2 + 2 * b * X * Y + c * Y**2))
    Kernel = (D2Gxx + D2Gyy) / np.sum(Gaussian.flatten())
    return Kernel
