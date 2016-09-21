import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as dtx
import scipy.ndimage.filters as filters


def ChanVese(I, Mask, Sigma, dt=1.0, Mu=0.2, Lambda1=1, Lambda2=1, It=100):
    """Region-based level sets.

    Region-based level set implementation based on the Chan-Vese method.
    Provides cost terms for boundary length, the variance of intensities inside
    the zero level-set, and the variance of external intensities. Robust to
    initialization.

    Parameters
    ----------
    I : array_like
        A floating-point intensity image.
    Mask : array_like
        A binary mask initializing the level set image. Regions corresponding
        to the interior of the level-set function have value 1, with other
        regions having value 0.
    Sigma : double
        Standard deviation of smoothing filter for input image I.
    dt : double
        Time step for evolving the level-set function Phi. Default value = 1.0.
    Mu : double
        Boundary length weight for energy function. Default value = 0.2.
    Lambda1 : double
        Internal variance weight for energy function. Default value = 1.
    Lambda2 : double
        External variance weight for energy function. Default value = 1.
    It : double
        Number of iterations to evolve curve level set function over. Default
        value = 100.


    Returns
    -------
    Phi : array_like
        An intensity image where the zero level set defines object boundaries.
        Can be further processed with fast marching methods or other to obtain
        smooth boundaries, or simply thresholded to define the object mask.

    See Also
    --------
    histomicstk.segmentation.nuclear.GaussianVoting

    References
    ----------
    .. [1] C. Li, C. Xu, C. Gui, M.D. fox, "Distance Regularized Level Set
           Evolution and Its Application to Image Segmentation," in IEEE
           Transactions on Image Processing, vol.19,no.12,pp.3243-54, 2010.
    """

    # smoothed gradient of input image
    I = filters.gaussian_filter(I, Sigma, mode='constant', cval=0)
#    dsI = np.gradient(sI)
#    I = 1/(1 + dsI[0]**2 + dsI[1]**2)

    # generate signed distance map
    Phi = ConvertMask(Mask)

    # evolve level set function
    for i in range(0, It):

        # calculate interior and exterior averages
        C1 = np.sum(I[Phi > 0]) / (np.sum(Phi > 0) + 1e-10)
        C2 = np.sum(I[Phi <= 0]) / (np.sum(Phi <= 0) + 1e-10)
        Force = Lambda2 * (I - C2)**2 - Lambda1 * (I - C1)**2

        # curvature of image
        Curvature = Kappa(Phi)

        # evolve
        Phi += dt * Force / np.max(np.abs(Force)) + Mu*Curvature

    return Phi


def ConvertMask(Mask):
    # convert binary mask to signed distance function
    Phi0 = dtx(1-Mask) - dtx(Mask) + Mask - 1/2
    return Phi0


def Kappa(Phi):
    dPhi = np.gradient(Phi)  # calculate gradient of level set image
    xdPhi = np.gradient(dPhi[1])
    ydPhi = np.gradient(dPhi[0])
    K = (xdPhi[1]*(dPhi[0]**2) - 2*xdPhi[0]*dPhi[0]*dPhi[1] +
         ydPhi[0]*(dPhi[1]**2)) / ((dPhi[0]**2 + dPhi[1]**2 + 1e-10)**(3/2))
    K *= (xdPhi[1]**2 + ydPhi[0]**2)**(1/2)
    return K


def Impulse(X, Epsilon):
    # Smooth dirac delta function.

    # calculate smoothed impulse everywhere
    Xout = (1 + np.cos(np.pi * X / Epsilon)) / (2 * Epsilon)

    # zero out values |x| > Epsilon
    Xout[np.absolute(X) > Epsilon] = 0

    return Xout
