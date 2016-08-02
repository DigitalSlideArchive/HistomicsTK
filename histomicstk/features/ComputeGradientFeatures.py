import numpy as np
from scipy.stats import entropy, skew, kurtosis
from skimage.feature import canny


def ComputeGradientFeatures(I, Coords, Diff=[], C=[]):
    """
    Calculates gradient features from an intensity image.

    Parameters
    ----------
    I : array_like
        A M x N intensity image.
    Coords : array_like
        A T x 2 coordinate list of a region.
    Diff : array_like
        A M x N gradient difference. Default value = [].
    C : array_like
        A M x N canny filtered mask. Default value = [].

    Returns
    -------
    MeanGradMag : float
        Mean of gradient data.
    StdGradMag : float
        Standard deviation of gradient data.
    EntropyGradMag : float
        Entroy of gradient data.
    EnergyGradMag : float
        Energy of gradient data.
    SkewnessGradMag : float
        Skewness of gradient data. Value is 0 when all values are equal.
    KurtosisGradMag : float
        Kurtosis of gradient data. Value is -3 when all values are equal.
    SumCanny : float
        Sum of canny filtered gradient data.
    MeanCanny : float
        Mean of canny filtered gradient data.

    References
    ----------
    .. [1] Daniel Zwillinger and Stephen Kokoska. "CRC standard probability
    and statistics tables and formulae," Crc Press, 1999.
    """

    if not len(Diff):
        Gx, Gy = np.gradient(I)
        Diff = np.sqrt(Gx**2 + Gy**2)

    if not len(C):
        C = canny(I)

    pixOfInterest = Diff[Coords[:, 0], Coords[:, 1]]
    # compute gradient features
    MeanGradMag = np.mean(pixOfInterest)
    StdGradMag = np.std(pixOfInterest)
    EntropyGradMag = entropy(pixOfInterest)
    hist, bins = np.histogram(pixOfInterest, bins=np.arange(256))
    prob = hist/np.sum(hist, dtype=np.float32)
    EnergyGradMag = np.sum(np.power(prob, 2))
    SkewnessGradMag = skew(pixOfInterest)
    KurtosisGradMag = kurtosis(pixOfInterest)
    bw_canny = C[Coords[:, 0], Coords[:, 1]]
    SumCanny = np.sum(bw_canny)
    MeanCanny = SumCanny/len(pixOfInterest)

    return MeanGradMag, StdGradMag, EntropyGradMag, EnergyGradMag, \
        SkewnessGradMag, KurtosisGradMag, SumCanny, MeanCanny
