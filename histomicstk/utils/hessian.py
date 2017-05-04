import numpy as np
from scipy.ndimage.filters import convolve


def hessian(im_input, sigma):
    """
    Calculates hessian of image I convolved with a gaussian kernel with
    covariance C = [Sigma^2 0; 0 Sigma^2].

    Parameters
    ----------
    im_input : array_like
        M x N grayscale image.
    sigma : double
        standard deviation of gaussian kernel.

    Returns
    -------
    im_hess : array_like
        M x N x 4 hessian matrix - im_hess[:,:,0] = dxx,
        im_hess[:,:,1] = im_hess[:,:,2] = dxy, im_hess[:,:,3] = dyy.

    """

    # generate kernel domain
    h, k = round(3 * sigma), round(3 * sigma + 1)
    x, y = np.mgrid[-h:k, -h:k]

    # generate kernels
    gxx = 1./(2 * np.pi * sigma ** 4) * ((x / sigma) ** 2 - 1) * \
        np.exp(-(x**2+y**2) / (2 * sigma ** 2))
    gxy = 1./(2 * np.pi * sigma ** 6) * np.multiply(x, y) * \
        np.exp(-(x**2+y**2) / (2 * sigma ** 2))
    gyy = np.transpose(gxx)

    # convolve
    dxx = convolve(im_input, gxx, mode='constant')
    dxy = convolve(im_input, gxy, mode='constant')
    dyy = convolve(im_input, gyy, mode='constant')

    # format output
    im_hess = np.concatenate(
        (dxx[:, :, None], dxy[:, :, None], dxy[:, :, None], dyy[:, :, None]),
        axis=2
    )
    return im_hess
