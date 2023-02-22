import numpy as np


def chan_vese(im_input, im_mask, sigma,
              dt=1.0, mu=0.2, lambda1=1, lambda2=1, iter=100):
    """Region-based level sets.

    Region-based level set implementation based on the Chan-Vese method.
    Provides cost terms for boundary length, the variance of intensities inside
    the zero level-set, and the variance of external intensities. Robust to
    initialization.

    Parameters
    ----------
    im_input : array_like
        A floating-point intensity image.
    im_mask : array_like
        A binary mask initializing the level set image. Regions corresponding
        to the interior of the level-set function have value 1, with other
        regions having value 0.
    sigma : double
        Standard deviation of smoothing filter for input image im_input.
    dt : double
        Time step for evolving the level-set function im_phi. Default value = 1.0.
    mu : double
        Boundary length weight for energy function. Default value = 0.2.
    lambda1 : double
        Internal variance weight for energy function. Default value = 1.
    lambda2 : double
        External variance weight for energy function. Default value = 1.
    iter : double
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
    import scipy.ndimage as ndi

    # smoothed gradient of input image
    im_input = ndi.gaussian_filter(im_input, sigma, mode='constant', cval=0)

    # generate signed distance map
    im_phi = mask_to_sdf(im_mask)

    # evolve level set function
    for i in range(0, iter):

        # calculate interior and exterior averages
        C1 = np.sum(im_input[im_phi > 0]) / (np.sum(im_phi > 0) + 1e-10)
        C2 = np.sum(im_input[im_phi <= 0]) / (np.sum(im_phi <= 0) + 1e-10)
        Force = lambda2 * (im_input - C2) ** 2 - lambda1 * (im_input - C1) ** 2

        # curvature of image
        Curvature = kappa(im_phi)

        # evolve
        im_phi += dt * Force / np.max(np.abs(Force)) + mu * Curvature

    return im_phi


def mask_to_sdf(im_mask):
    from scipy.ndimage import distance_transform_edt as dtx

    # convert binary mask to signed distance function

    Phi0 = dtx(1 - im_mask) - dtx(im_mask) + im_mask - 1 / 2
    return Phi0


def kappa(im_phi):

    dPhi = np.gradient(im_phi)  # calculate gradient of level set image
    xdPhi = np.gradient(dPhi[1])
    ydPhi = np.gradient(dPhi[0])
    K = (xdPhi[1] * (dPhi[0]**2) - 2 * xdPhi[0] * dPhi[0] * dPhi[1] +
         ydPhi[0] * (dPhi[1]**2)) / ((dPhi[0]**2 + dPhi[1]**2 + 1e-10)**(3 / 2))
    K *= (xdPhi[1]**2 + ydPhi[0]**2)**(1 / 2)
    return K
