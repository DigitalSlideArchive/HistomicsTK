import numpy as np
from scipy.stats import entropy, skew, kurtosis

def ComputeTextureFeatures(I, Coords):
    """
    Calculates texture features from an intensity image.

    Parameters
    ----------
    I : array_like
        A M x N intensity image.
    Coords : array_like
        A T x 2 coordinate list of a region.

    Returns
    -------
    Entropy : float
        Entroy of intensity data.
    Energy : float
        Energy of intensity data.
    Skewness : float
        Skewness of intensity data. Value is 0 when all values are equal.
    Kurtosis : float
        Kurtosis of intensity data. Value is -3 when all values are equal.

    References
    ----------
    .. [1] Daniel Zwillinger and Stephen Kokoska. "CRC standard probability
       and statistics tables and formulae," Crc Press, 1999.
    """

    # compute texture features
    Entropy = entropy(I[Coords[:,0], Coords[:,1]])
    hist, bins = np.histogram(I[Coords[:, 0], Coords[:, 1]],
        bins=np.arange(256))
    prob = hist/np.sum(hist, dtype=np.float32)
    Energy = np.sum(np.power(prob, 2))
    Skewness = skew(I[Coords[:,0], Coords[:,1]])
    Kurtosis = kurtosis(I[Coords[:,0], Coords[:,1]])

    return Entropy, Energy, Skewness, Kurtosis
