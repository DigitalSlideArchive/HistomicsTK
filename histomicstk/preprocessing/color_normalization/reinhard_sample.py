import collections
import numpy as np
from histomicstk.utils import sample_pixels
from histomicstk.preprocessing import color_conversion


def reinhard_sample(slide_path, magnification, sample_percent, tile_size):
    """Samples a whole-slide-image to determine sample_pix_lab colorspace
    statistics (mean, variance) needed to perform global Reinhard color
    normalization.

    Normalizing individual tiles independently creates a significant bias
    in the results of segmentation and feature extraction, as the color
    statistics of each tile in a whole-slide image can vary significantly.
    To remedy this, we sample a subset of pixels from the entire whole-slide
    image in order to estimate the global mean and standard deviation of
    each channel in the Lab color space that are needed for reinhard color
    normalization.

    Parameters
    ----------
    slide_path : str
        path and filename of slide.
    magnification : scalar
        Desired magnification for sampling. Defaults to native scan
        magnification.
    sample_percent : double
       Percentage of pixels to sample (range (0, 1]).
    tile_size : int
       tile_size size used in sampling high-resolution image.

    Returns
    -------
    Mu : array_like
        A 3-element array containing the means of the target image channels
        in sample_pix_lab color space.
    Sigma : array_like
        A 3-element list containing the standard deviations of the target image
        channels in sample_pix_lab color space.

    Notes
    -----
    Returns a namedtuple.

    See Also
    --------
    histomicstk.preprocessing.color_conversion.lab_mean_std
    histomicstk.preprocessing.color_normalization.reinhard
    """

    # generate a sampling of sample_pixels_rgb pixels from whole-slide image
    sample_pixels_rgb = sample_pixels(slide_path, magnification,
                                      sample_percent, tile_size)

    # reshape the 3xN pixel array into a 1 x N x 3 image for lab_mean_std
    sample_pixels_rgb = np.reshape(sample_pixels_rgb.transpose(),
                                   (1, sample_pixels_rgb.shape[1], 3))

    # compute mean and stddev of sample pixels in Lab space
    Mu, Sigma = color_conversion.lab_mean_std(sample_pixels_rgb)

    # build named tuple for output
    ReinhardStats = collections.namedtuple('ReinhardStats', ['Mu', 'Sigma'])
    stats = ReinhardStats(Mu, Sigma)

    return stats
