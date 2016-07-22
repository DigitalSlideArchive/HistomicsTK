import collections
import numpy as np
from histomicstk.utils import Sample
from histomicstk.preprocessing.color_conversion import RudermanLABFwd


def ReinhardSample(File, Magnification, Percent, Tile):
    """Samples a whole-slide-image to determine LAB colorspace statistics
    (mean, variance) needed to perform global Reinhard color normalization.

    Normalizing individual tiles independently creates a significant bias
    in the results of segmentation and feature extraction, as the color
    statistics of each tile in a whole-slide image can vary significantly.
    To remedy this, we sample from the entire whole-slide image in order to
    obtain the global mean and variance of the LAB colorspace channels that
    can then be used when processing individual tiles for uniformity. This
    function can also be used to obtain the global target statistics from
    an ideal slide that can serve as the normalization standard.

    Parameters
    ----------
    File : str
        path and filename of slide.
    Magnification : scalar
        Desired magnification for sampling. Defaults to native scan
        magnification.
    Percent : double
       Percentage of pixels to sample (range (0, 1]).
    Tile : int
       Tile size used in sampling high-resolution image.

    Returns
    -------
    TargetMu : array_like
        A 3-element array containing the means of the target image channels
        in LAB color space.
    TargetSigma : array_like
        A 3-element list containing the standard deviations of the target image
        channels in LAB color space.

    Notes
    -----
    Returns a namedtuple.

    See Also
    --------
    histomicstk.preprocessing.color_conversion.RudermanLABFwd,
    histomicstk.preprocessing.color_conversion.RudermanLABInv
    """

    # generate a sampling of RGB pixels from whole-slide image
    RGB = Sample(File, Magnification, Percent, Tile)

    # reshape the 3xN pixel array into an image for RudermanLABFwd
    RGB = np.reshape(RGB.transpose(), (1, RGB.shape[1], 3))

    # perform forward LAB transformation
    LAB = RudermanLABFwd(RGB)

    # compute statistics of LAB channels
    Mu = LAB.sum(axis=0).sum(axis=0) / (LAB.size / 3)
    LAB[:, :, 0] = LAB[:, :, 0] - Mu[0]
    LAB[:, :, 1] = LAB[:, :, 1] - Mu[1]
    LAB[:, :, 2] = LAB[:, :, 2] - Mu[2]
    Sigma = ((LAB * LAB).sum(axis=0).sum(axis=0) / (LAB.size / 3 - 1)) ** 0.5

    # build named tuple for output
    OutTuple = collections.namedtuple('Statistics', ['Mu', 'Sigma'])
    Output = OutTuple(Mu, Sigma)

    return Output
