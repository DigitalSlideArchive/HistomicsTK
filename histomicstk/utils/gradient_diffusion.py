import numpy as np
import scipy.ndimage.filters as spf


def gradient_diffusion(dX, dY, Mask, Mu=5, Lambda=5, Iterations=10,
                       dT=0.05):
    """
    Diffusion of gradient field using Navier-Stokes equation. Used for
    smoothing/denoising a gradient field.

    Takes as input a gradient field image (dX, dY), and a mask of the
    foreground region, and then iteratively solves the Navier-Stokes equation
    to diffuse the vector field and align noisy gradient vectors with their
    surrounding signals.

    Parameters
    ----------
    dX : array_like
        Horizontal component of gradient image.
    dY : array_like
        Vertical component of gradient image.
    Mu : float
        Weight parmeter from Navier-Stokes equation - weights divergence and
        Laplacian terms. Default value = 5.
    Lambda : float
        Weight parameter from Navier-Stokes equation - used to weight
        divergence. Default value = 5.
    Mask : array_like
        Binary mask where foreground objects have value 1, and background
        objects have value 0. Used to restrict influence of background vectors
        on diffusion process.
    Iterations : float
        Number of time-steps to use in solving Navier-Stokes. Default value =
        10.
    dT : float
        Timestep to be used in solving Navier-Stokes. Default value = 0.05.

    Returns
    -------
    vX : array_like
        Horizontal component of diffused gradient.
    vY : array_like
        Vertical component of diffused gradient.

    See Also
    --------
    histomicstk.segmentation.nuclear.GradientFlow

    References
    ----------
    .. [1] G. Li et al "3D cell nuclei segmentation based on gradient flow
           tracking" in BMC Cell Biology,vol.40,no.8, 2007.
    """

    # initialize solution
    vX = dX.copy()
    vY = dY.copy()

    # iterate for prescribed number of iterations
    for it in range(Iterations):

        # calculate divergence of current solution
        vXY, vXX = np.gradient(vX)
        vYY, vYX = np.gradient(vY)
        Div = vXX + vYY
        DivY, DivX = np.gradient(Div)

        # calculate laplacians of current solution
        vX += dT*(Mu * spf.laplace(vX) + (Lambda + Mu) * DivX + Mask *
                  (dX - vX))
        vY += dT*(Mu * spf.laplace(vY) + (Lambda + Mu) * DivY + Mask *
                  (dY - vY))

    # return solution
    return vX, vY
