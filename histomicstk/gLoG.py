import numpy as np
import scipy.ndimage as ndi
from skimage import morphology


def gLoG(I, Alpha=1, Range=np.linspace(1.5, 3, np.round((3-1.5)/0.2)+1),
         Theta=np.pi/4, Tau=0.6, Eps=0.6):
    """Performs generalized Laplacian of Gaussian blob detection.

    Parameters
    ----------
    I : array_like
        A hematoxylin intensity image obtained from ColorDeconvolution.
    Alpha : double
        A positive scalar used to normalize the gLoG filter responses. Controls
        the blob-center detection and eccentricities of detected blobs. Larger
        values emphasize more eccentric blobs. Default value = 1.
    Range : array_like
    Theta : double
        Angular increment for rotating gLoG filters. Default value = np.pi / 6.
    Tau : double
        Tolerance for counting pixels in determining optimal scale SigmaC
    Eps : double
        Range to define SigmaX surrounding SigmaC

    Returns
    -------


    Notes
    -----
    Return values are returned as a namedtuple

    See Also
    --------
    ColorDeconvolution

    References
    ----------
    .. [1] H. Kong, H.C. Akakin, S.E. Sarma, "A Generalized Laplacian of
           Gaussian Filter for Blob Detection and Its Applications," in IEEE
           Transactions on Cybernetics, vol.43,no.6,pp.1719-33, 2013.
    """

    # initialize sigma
    Sigma = np.exp(Range)

    # generate circular LoG scale-space to determine range of SigmaX
    l_g = 0
    H = []
    Bins = []
    Min = np.zeros((len(Sigma), 1))
    Max = np.zeros((len(Sigma), 1))
    for i, s in enumerate(Sigma):
        Response = s**2 * ndi.filters.gaussian_laplace(I, s, output=None,
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
        Zeta[i] = np.sum(H[i][Bins[i][0:-1] > Tau])

    # identify best scale SigmaC based on maximum circular response
    Index = np.argmax(Zeta)

    # define range for SigmaX
    XRange = range(max(Index-2, 0), min(len(Range), Index+2)+1)
    SigmaX = np.exp(Range[XRange])

    # define rotation angles
    Thetas = np.linspace(0, np.pi-Theta, np.round(np.pi/Theta))

    # loop over SigmaX, SigmaY and then angle, summing up filter responses
    Rsum = np.zeros(I.shape)
    for i, Sx in enumerate(SigmaX):
        YRange = range(0, XRange[i])
        SigmaY = np.exp(Range[YRange])
        for Sy in SigmaY:
            for Th in Thetas:
                Kernel = gLoGKernel(Sx, Sy, Th)
                Kernel = (1 + np.log(Sx)**Alpha) * (1 + np.log(Sy)**Alpha) *\
                    Kernel
                Rsum += ndi.convolve(I, Kernel, mode='constant', cval=0.0)
                print(Sx, Sy, Th)
        Kernel = gLoGKernel(Sx, Sx, 0)
        Kernel = (1 + np.log(Sx)**Alpha) * (1 + np.log(Sx)**Alpha) * Kernel
        Rsum += ndi.convolve(I, Kernel, mode='constant', cval=0.0)
        print(Sx, Sx, 0)

    # detect local maxima
    Disk = morphology.disk(3*np.exp(Range[Index]))
    Maxima = ndi.filters.maximum_filter(Rsum, footprint=Disk)
    Maxima = Rsum == Maxima

    return Rsum, Maxima


def gLoGKernel(SigmaX, SigmaY, Theta):
    N = np.ceil(2 * 3 * SigmaX)
    X, Y = np.meshgrid(np.linspace(0, N, N + 1) - N / 2,
                       np.linspace(0, N, N + 1) - N / 2)
    a = np.cos(Theta)**2 / (2 * SigmaX**2) + np.sin(Theta)**2 / (2 * SigmaY**2)
    b = -np.sin(2 * Theta) / (4 * SigmaX**2) + np.sin(2 * Theta) /\
        (4 * SigmaY**2)
    c = np.sin(Theta)**2 / (2 * SigmaX**2) + np.cos(Theta)**2 / (2 * SigmaY**2)
    D2Gxx = ((2*a*X + 2*b*Y)**2 - 2*a) * np.exp(-(a*X**2 + 2*b*X*Y + c*Y**2))
    D2Gyy = ((2*b*X + 2*c*Y)**2 - 2*c) * np.exp(-(a*X**2 + 2*b*X*Y + c*Y**2))
    Gaussian = np.exp(-(a*X**2 + 2*b*X*Y + c*Y**2))
    Kernel = (D2Gxx + D2Gyy) / np.sum(Gaussian.flatten())
    return Kernel
