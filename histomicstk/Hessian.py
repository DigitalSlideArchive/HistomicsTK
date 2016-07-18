import numpy as np
from scipy.ndimage.filters import convolve


def Hessian(I, Sigma):
    """
    Calculates hessian of image I convolved with a gaussian kernel with
    covariance C = [Sigma^2 0; 0 Sigma^2].
    Parameters
    ----------
    I : array_like
        M x N grayscale image.
    Sigma : double
        standard deviation of gaussian kernel.
    Returns
    -------
    H - array_like
        M x N x 4 hessian matrix - H[:,:,0] = dxx,
        H[:,:,1] = H[:,:,2] = dxy, H[:,:,3] = dyy.
    """

    # generate kernel domain
    h, k = round(3*Sigma), round(3*Sigma+1)
    x, y = np.mgrid[-h:k, -h:k]

    # generate kernels
    gxx = 1./(2*np.pi*Sigma**4)*((x/Sigma)**2-1) * \
        np.exp(-(x**2+y**2)/(2*Sigma**2))
    gxy = 1./(2*np.pi*Sigma**6)*np.multiply(x, y) * \
        np.exp(-(x**2+y**2)/(2*Sigma**2))
    gyy = np.transpose(gxx)

    # convolve
    dxx = convolve(I, gxx, mode='constant')
    dxy = convolve(I, gxy, mode='constant')
    dyy = convolve(I, gyy, mode='constant')

    # format output
    H = np.concatenate(
        (dxx[:, :, None], dxy[:, :, None], dxy[:, :, None], dyy[:, :, None]),
        axis=2
    )

    return H
