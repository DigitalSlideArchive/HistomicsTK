import numpy as np

import histomicstk.utils as htk_utls


def reg_edge(im_input, im_phi, well='double', sigma=1.5, dt=1.0, mu=0.2,
             lamda=1, alpha=-3, epsilon=1.5, iter=100):
    """Distance-regularized edge-based level sets.

    Distance-regularization is used in this edge-based level set implementation
    to avoid numerical problems requiring costly re-initialization. Provides
    cost terms for boundary length, area, and regularization of the level set
    function. Foreground objects are assumed to have larger intensity values
    than background.

    Parameters
    ----------
    im_input : array_like
        A floating-point intensity image.
    im_phi : array_like
        A floating-point initalization of the level-set image. Interior values
        are set to -c0, and exterior values set to c0, where c0 > 0.
    well : string
        Choice of well function for regularization. Can be set to either
        'single' or 'double' for single-well or double-well regularization, or
        any other value for no regularization. Default value = 'double'.
    sigma : double
        Standard deviation of smoothing filter for input image im_input.
    dt : double
        Time step for evolving im_phi. Default value = 1.0.
    mu : double
        Regularization weight for energy function. Default value = 0.2.
    lamda : double
        Boundary length weight for energy function. Default value = 1.0.
    alpha : double
        Area weight for energy function. A negative value is used to seed the
        interior of the foreground objects and then evolve the boundary
        outwards. A positive value assumes that the boundary begins outside the
        foreground objects and collapses to their high-gradient edges.
        Default value = -3.
    epsilon: double
        Coefficient used to smooth the Dirac and Heaviside functions. Default
        value = 1.5.
    iter: double
        Number of iterations to evolve curve level set function over. Default
        value = 100.

    Returns
    -------
    im_phi : array_like
        An intensity image where the zero level set defines object boundaries.
        Can be further processed with fast marching methods or other to obtain
        smooth boundaries, or simply thresholded to define the object mask.

    See Also
    --------
    histomicstk.segmentation.nuclear.gaussian_voting

    References
    ----------
    .. [#] C. Li, C. Xu, C. Gui, M.D. Fox, "Distance Regularized Level Set
       Evolution and Its Application to Image Segmentation," in IEEE
       Transactions on Image Processing, vol.19,no.12,pp.3243-54, 2010.

    """
    import scipy.ndimage.filters as filters

    # smoothed gradient of input image
    sI = filters.gaussian_filter(im_input, sigma, mode='constant', cval=0)
    dsI = np.gradient(sI)
    G = 1/(1 + dsI[0]**2 + dsI[1]**2)
    dG = np.gradient(G)

    # perform regularized level-set evolutions with time step dt
    for i in range(0, iter):

        # fix boundary conditions
        im_phi = neumann_bounds(im_phi)

        # calculate gradient of level set image
        dPhi = np.gradient(im_phi)
        mPhi = (dPhi[0]**2 + dPhi[1]**2)**0.5  # gradient magnitude
        Curve = np.gradient(dPhi[0] / (mPhi + 1e-10))[0] + \
            np.gradient(dPhi[1] / (mPhi + 1e-10))[1]  # divergence

        # build regularization function
        if well == 'single':
            Reg = single_well(im_phi, Curve)
        elif well == 'double':
            Reg = double_well(im_phi, dPhi, mPhi, Curve, i)
        else:
            Reg = np.zeros(im_phi.shape)

        # area and boundary-length energy function terms
        iPhi = impulse(im_phi, epsilon)
        Area = iPhi * G
        Edge = iPhi * (dG[0] * (dPhi[0] / (mPhi + 1e-10)) +
                       dG[1] * (dPhi[1] / (mPhi + 1e-10))) + iPhi * G * Curve

        # evolve level-set function
        im_phi = im_phi + dt * (mu * Reg + lamda * Edge + alpha * Area)

    # return evolved level-set function following iterations
    return im_phi


def initialize(Mask, c0=2):
    # initialize scaled binary-step image
    Phi0 = np.zeros(Mask.shape)
    Phi0[Mask > 0] = -c0
    Phi0[Mask == 0] = c0
    return Phi0


def single_well(Phi, Curve):
    # Single-well potential function
    return 4 * htk_utls.del2(Phi)-Curve


def double_well(Phi, dPhi, mPhi, Curve, i):
    # Double-well potential function
    SmallMask = (mPhi <= 1) & (mPhi >= 0)
    LargeMask = (mPhi > 1)
    P = SmallMask * np.sin(2 * np.pi * mPhi) / \
        (2 * np.pi) + LargeMask * (mPhi - 1)
    dP = ((P != 0) * P + (P == 0)) / ((mPhi != 0) * mPhi + (mPhi == 0))
    Well = np.gradient(dP * dPhi[0] - dPhi[0])[0] + \
        np.gradient(dP * dPhi[1] - dPhi[1])[1] + 4 * htk_utls.del2(Phi)
    return Well


def impulse(X, Epsilon):
    # Smooth dirac delta function.

    # calculate smoothed impulse everywhere
    Xout = (1 + np.cos(np.pi * X / Epsilon)) / (2 * Epsilon)

    # zero out values |x| > Epsilon
    Xout[np.absolute(X) > Epsilon] = 0

    return Xout


def neumann_bounds(Phi):
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
