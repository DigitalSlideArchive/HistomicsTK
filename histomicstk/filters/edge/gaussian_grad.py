import collections

import numpy as np


def gaussian_grad(im_input, sigma):
    """Performs smoothing with derivative gaussian kernel.

    Uses seperable convolution to simultaneously smooth and calculate the
    gradient of a grayscale image.

    Parameters
    ----------
    im_input : array_like
        An intensity image.
    sigma : double
        Standard deviation of smoothing kernel used in gradient calculation.

    Returns
    -------
    dx : array_like
        An intensity image of the X gradient component.
    dy : array_like
        An intensity image of the Y gradient component.

    Notes
    -----
    Return values are returned as a namedtuple

    """
    import scipy.signal as signal

    # generate separable gaussian derivative kernels
    x = np.linspace(0, np.ceil(2 * 3 * sigma), int(np.ceil(2 * 3 * sigma) + 1))
    y = np.linspace(0, np.ceil(2 * 3 * sigma), int(np.ceil(2 * 3 * sigma) + 1))
    x -= np.ceil(2 * 3 * sigma) / 2  # center independent variables at zero
    y -= np.ceil(2 * 3 * sigma) / 2
    x = np.reshape(x, (1, x.size))  # reshape to 2D row and column vectors
    y = np.reshape(y, (y.size, 1))
    xGx = 2 * x / (sigma ** 2) * np.exp(-x ** 2 / (2 * sigma ** 2)) \
        / (sigma * (2 * np.pi) ** 0.5)
    yGx = np.exp(-y**2 / (2 * sigma ** 2)) / ((2 * np.pi) ** 0.5 * sigma)
    xGy = np.exp(-x**2 / (2 * sigma ** 2)) / ((2 * np.pi) ** 0.5 * sigma)
    yGy = 2 * y / (sigma ** 2) * np.exp(-y ** 2 / (2 * sigma ** 2)) \
        / (sigma * (2 * np.pi) ** 0.5)

    # smoothed gradients of input image
    dx = signal.convolve2d(im_input, xGx, mode='same')
    dx = signal.convolve2d(dx, yGx, mode='same')
    dy = signal.convolve2d(im_input, xGy, mode='same')
    dy = signal.convolve2d(dy, yGy, mode='same')

    # format output
    Output = collections.namedtuple('Output', ['dx', 'dy'])
    Gradients = Output(dx, dy)

    return Gradients
