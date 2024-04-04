import collections

import numpy as np

from histomicstk.preprocessing import color_conversion
from histomicstk.utils import sample_pixels


def reinhard_stats(
        slide_path,
        sample_fraction,
        magnification=None,
        tissue_seg_mag=1.25,
        invert_image=False,
        style=None,
        frame=None,
        default_img_inversion=False):
    """Samples a whole-slide-image to determine colorspace statistics (mean,
    variance) needed to perform global Reinhard color normalization.

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
    sample_fraction : double
       Fraction of pixels to sample (range (0, 1]).
    magnification : scalar
        Desired magnification for sampling. Defaults to native scan
        magnification.
    tissue_seg_mag: double, optional
        low resolution magnification at which foreground will be segmented.
        Default value = 1.25.

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
    sample_pixels_rgb = sample_pixels(
        slide_path,
        sample_fraction=sample_fraction,
        magnification=magnification,
        tissue_seg_mag=tissue_seg_mag,
        invert_image=invert_image,
        style=style,
        frame=frame,
        default_img_inversion=default_img_inversion,
    )

    # reshape the Nx3 pixel array into a 1 x N x 3 image for lab_mean_std
    sample_pixels_rgb = np.reshape(sample_pixels_rgb,
                                   (1, sample_pixels_rgb.shape[0], 3))

    # compute mean and stddev of sample pixels in Lab space
    Mu, Sigma = color_conversion.lab_mean_std(sample_pixels_rgb)

    # build named tuple for output
    ReinhardStats = collections.namedtuple('ReinhardStats', ['Mu', 'Sigma'])
    stats = ReinhardStats(Mu, Sigma)

    return stats


def reinhard_stats_rgb(rgb_image):
    rgb_pixels = np.reshape(rgb_image, (1, rgb_image.shape[0] * rgb_image.shape[1], 3))

    Mu, Sigma = color_conversion.lab_mean_std(rgb_pixels)

    # build named tuple for output
    return collections.namedtuple('ReinhardStats', ['Mu', 'Sigma'])(Mu, Sigma)
