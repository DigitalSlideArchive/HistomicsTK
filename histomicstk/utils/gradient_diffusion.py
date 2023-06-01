import numpy as np


def gradient_diffusion(im_dx, im_dy, im_fgnd_mask,
                       mu=5, lambda_=5, iterations=10, dt=0.05):
    """
    Diffusion of gradient field using Navier-Stokes equation. Used for
    smoothing/denoising a gradient field.

    Takes as input a gradient field image (dX, dY), and a mask of the
    foreground region, and then iteratively solves the Navier-Stokes equation
    to diffuse the vector field and align noisy gradient vectors with their
    surrounding signals.

    Parameters
    ----------
    im_dx : array_like
        Horizontal component of gradient image.
    im_dy : array_like
        Vertical component of gradient image.
    im_fgnd_mask : array_like
        Binary mask where foreground objects have value 1, and background
        objects have value 0. Used to restrict influence of background vectors
        on diffusion process.
    mu : float
        Weight parameter from Navier-Stokes equation - weights divergence and
        Laplacian terms. Default value = 5.
    lambda_ : float
        Weight parameter from Navier-Stokes equation - used to weight
        divergence. Default value = 5.
    iterations : float
        Number of time-steps to use in solving Navier-Stokes. Default value =
        10.
    dt : float
        Timestep to be used in solving Navier-Stokes. Default value = 0.05.

    Returns
    -------
    im_vx : array_like
        Horizontal component of diffused gradient.
    im_vy : array_like
        Vertical component of diffused gradient.

    See Also
    --------
    histomicstk.segmentation.nuclear.GradientFlow

    References
    ----------
    .. [#] G. Li et al "3D cell nuclei segmentation based on gradient flow
           tracking" in BMC Cell Biology,vol.40,no.8, 2007.

    """
    import scipy.ndimage as ndi

    # initialize solution
    im_vx = im_dx.copy()
    im_vy = im_dy.copy()

    # iterate for prescribed number of iterations
    for it in range(iterations):

        # calculate divergence of current solution
        vXY, vXX = np.gradient(im_vx)
        vYY, vYX = np.gradient(im_vy)
        Div = vXX + vYY
        DivY, DivX = np.gradient(Div)

        # calculate laplacians of current solution
        im_vx += dt * (mu * ndi.laplace(im_vx) +
                       (lambda_ + mu) * DivX +
                       im_fgnd_mask * (im_dx - im_vx))
        im_vy += dt * (mu * ndi.laplace(im_vy) +
                       (lambda_ + mu) * DivY +
                       im_fgnd_mask * (im_dy - im_vy))

    # return solution
    return im_vx, im_vy
