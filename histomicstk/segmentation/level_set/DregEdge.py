import histomicstk.utils as htk_utls
import numpy as np
import scipy.ndimage.filters as filters


def DregEdge(I, Phi, Well='double', Sigma=1.5, dt=1.0, Mu=0.2, Lambda=1,
             Alpha=-3, Epsilon=1.5, It=100):
    """Distance-regularized edge-based level sets.

    Distance-regularization is used in this edge-based level set implementation
    to avoid numerical problems requiring costly re-initialization. Provides
    cost terms for boundary length, area, and regularization of the level set
    function. Foreground objects are assumed to have larger intensity values
    than background.

    Parameters
    ----------
    I : array_like
        A floating-point intensity image.
    Phi : array_like
        A floating-point initalization at the level-set image. Interior values
        are set to -c0, and exterior values set to c0, where c0 > 0.
    Well : string
        Choice of well function for regularization. Can be set to either
        'single' or 'double' for single-well or double-well regularization, or
        any other value for no regularization. Default value = 'double'.
    Sigma : double
        Standard deviation of smoothing filter for input image I.
    dt : double
        Time step for evolving Phi. Default value = 1.0.
    Mu : double
        Regularization weight for energy function. Default value = 0.2.
    Lambda : double
        Boundary length weight for energy function. Default value = 1.0.
    Alpha : double
        Area weight for energy function. A negative value is used to seed the
        interior of the foreground objects and then evolve the boundary
        outwards. A positive value assumes that the boundary begins outside the
        foreground objects and collapses to their high-gradient edges.
        Default value = -3.
    Epsilon: double
        Coefficient used to smooth the Dirac and Heaviside functions. Default
        value = 1.5.
    It: double
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
    GaussianVoting

    References
    ----------
    .. [1] C. Li, C. Xu, C. Gui, M.D. fox, "Distance Regularized Level Set
           Evolution and Its Application to Image Segmentation," in IEEE
           Transactions on Image Processing, vol.19,no.12,pp.3243-54, 2010.
    """

    # smoothed gradient of input image
    sI = filters.gaussian_filter(I, Sigma, mode='constant', cval=0)
    dsI = np.gradient(sI)
    G = 1/(1 + dsI[0]**2 + dsI[1]**2)
    dG = np.gradient(G)

    # perform regularized level-set evolutions with time step dt
    for i in range(0, It):

        # fix boundary conditions
        Phi = NeumannBounds(Phi)

        # calculate gradient of level set image
        dPhi = np.gradient(Phi)
        mPhi = (dPhi[0]**2 + dPhi[1]**2)**0.5  # gradient magnitude
        Curve = np.gradient(dPhi[0] / (mPhi + 1e-10))[0] + \
            np.gradient(dPhi[1] / (mPhi + 1e-10))[1]  # divergence

        # build regularization function
        if Well == 'single':
            Reg = SingleWell(Phi, Curve)
        elif Well == 'double':
            Reg = DoubleWell(Phi, dPhi, mPhi, Curve, i)
        else:
            Reg = np.zeros(Phi.shape)

        # area and boundary-length energy function terms
        iPhi = Impulse(Phi, Epsilon)
        Area = iPhi * G
        Edge = iPhi * (dG[0] * (dPhi[0] / (mPhi + 1e-10)) +
                       dG[1] * (dPhi[1] / (mPhi + 1e-10))) + iPhi * G * Curve

        # evolve level-set function
        Phi = Phi + dt * (Mu * Reg + Lambda * Edge + Alpha * Area)

    # return evolved level-set function following iterations
    return Phi


def Initialize(Mask, c0=2):
    # initialize scaled binary-step image
    Phi0 = np.zeros(Mask.shape)
    Phi0[Mask > 0] = -c0
    Phi0[Mask == 0] = c0
    return Phi0


def SingleWell(Phi, Curve):
    # Single-well potential function
    return 4 * htk_utls.Del2(Phi)-Curve


def DoubleWell(Phi, dPhi, mPhi, Curve, i):
    # Double-well potential function
    SmallMask = (mPhi <= 1) & (mPhi >= 0)
    LargeMask = (mPhi > 1)
    P = SmallMask * np.sin(2 * np.pi * mPhi) / \
        (2 * np.pi) + LargeMask * (mPhi - 1)
    dP = ((P != 0) * P + (P == 0)) / ((mPhi != 0) * mPhi + (mPhi == 0))
    Well = np.gradient(dP * dPhi[0] - dPhi[0])[0] + \
        np.gradient(dP * dPhi[1] - dPhi[1])[1] + 4 * htk_utls.Del2(Phi)
    return Well


def Impulse(X, Epsilon):
    # Smooth dirac delta function.

    # calculate smoothed impulse everywhere
    Xout = (1 + np.cos(np.pi * X / Epsilon)) / (2 * Epsilon)

    # zero out values |x| > Epsilon
    Xout[np.absolute(X) > Epsilon] = 0

    return Xout


def NeumannBounds(Phi):
    # Transofrm input to enforce Neumann boundary conditions.

    # copy input
    PhiOut = Phi

    # capture image size
    m = Phi.shape[0]
    n = Phi.shape[1]

    # deal with corners
    PhiOut[0, 0] = PhiOut[2, 2]
    PhiOut[0, n-1] = PhiOut[0, -3]
    PhiOut[m-1, 0] = PhiOut[-3, 2]
    PhiOut[m-1, n-1] = PhiOut[-3, -3]

    # deal with edges
    PhiOut[0, 1:-1] = PhiOut[2, 1:-1]
    PhiOut[m-1, 1:-1] = PhiOut[m-3, 1:-1]
    PhiOut[1:-1, 0] = PhiOut[1:-1, 2]
    PhiOut[1:-1, n-1] = PhiOut[1:-1, n-3]

    return PhiOut
