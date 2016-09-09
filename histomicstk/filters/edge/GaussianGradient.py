import collections
import numpy as np
import scipy.signal as signal


def GaussianGradient(I, Sigma):
    """Performs smoothing with derivative gaussian kernel.

    Uses seperable convolution to simultaneously smooth and calculate the
    gradient of a grayscale image.

    Parameters
    ----------
    I : array_like
        An intensity image.
    sSigma : double
        Standard deviation of smoothing kernel used in gradient calculation.

    Returns
    -------
    dX : array_like
        An intensity image of the X gradient component.
    dY : array_like
        An intensity image of the Y gradient component.

    Notes
    -----
    Return values are returned as a namedtuple

    """

    # generate separable gaussian derivative kernels
    x = np.linspace(0, 2 * 3 * Sigma, 2 * 3 * Sigma + 1)
    y = np.linspace(0, 2 * 3 * Sigma, 2 * 3 * Sigma + 1)
    x -= 2 * 3 * Sigma / 2  # center independent variables at zero
    y -= 2 * 3 * Sigma / 2
    x = np.reshape(x, (1, x.size))  # reshape to 2D row and column vectors
    y = np.reshape(y, (y.size, 1))
    xGx = 2 * x / (Sigma**2) * np.exp(-x**2 / (2 * Sigma**2)) \
        / (Sigma * (2 * np.pi) ** 0.5)
    yGx = np.exp(-y**2 / (2 * Sigma**2)) / ((2 * np.pi) ** 0.5 * Sigma)
    xGy = np.exp(-x**2 / (2 * Sigma**2)) / ((2 * np.pi) ** 0.5 * Sigma)
    yGy = 2 * y / (Sigma**2) * np.exp(-y**2 / (2 * Sigma**2)) \
        / (Sigma * (2 * np.pi) ** 0.5)

    # smoothed gradients of input image
    dX = signal.convolve2d(I, xGx, mode='same')
    dX = signal.convolve2d(dX, yGx, mode='same')
    dY = signal.convolve2d(I, xGy, mode='same')
    dY = signal.convolve2d(dY, yGy, mode='same')

    # format output
    Output = collections.namedtuple('Output', ['dX', 'dY'])
    Gradients = Output(dX, dY)

    return Gradients
