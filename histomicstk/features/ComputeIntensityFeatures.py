import numpy as np
from scipy.stats import entropy, skew, kurtosis

def ComputeIntensityFeatures(I, Coords):
    """
    Calculates intensity features from an intensity image.

    Parameters
    ----------
    I : array_like
        A T x T intensity image.
    Coords : array_like
        A N x 2 coordinate list of a region.

    Returns
    -------
    IGroup : array_like
        A 1 x 9 intensity features.

    - `Intensity features`:
    MeanIntensity : Mean of intensity data.
    MeanMedianDifferenceIntensity : Difference between mean and median.
    MaxIntensity : Max intensity data.
    MinIntensity : Min intensity data.
    StdIntensity : Standard deviation of intensity data.
    Entropy : Entroy of intensity data.
    Energy : Energy of intensity data.
    Skewness : Skewness of intensity data.
        Value is 0 when all values are equal.
    Kurtosis : Kurtosis of intensity data.
        Value is -3 when all values are equal.
    """

    IGroup = np.zeros(9)
    pixOfInterest = I[Coords[:, 0], Coords[:, 1]]
    # compute mean
    IGroup[0] = np.mean(pixOfInterest)
    # compute mean median differnece
    IGroup[1] = IGroup[0] - np.median(pixOfInterest)
    # compute max
    IGroup[2] = max(pixOfInterest)
    # compute min
    IGroup[3] = min(pixOfInterest)
    # compute standard
    IGroup[4] = np.std(pixOfInterest)
    # compute entropy
    IGroup[5] = entropy(pixOfInterest)
    # compute energy
    hist, bins = np.histogram(pixOfInterest, bins=np.arange(256))
    prob = hist/np.sum(hist, dtype=np.float32)
    IGroup[6] = np.sum(np.power(prob, 2))
    # compute skewness
    IGroup[7] = skew(pixOfInterest)
    # compute kurtosis
    IGroup[8] = kurtosis(pixOfInterest)

    return IGroup
