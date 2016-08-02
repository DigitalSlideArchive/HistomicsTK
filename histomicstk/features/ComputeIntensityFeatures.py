import numpy as np

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
    MeanIntensity : float
        Mean of intensity data.
    MeanMedianDifferenceIntensity : float
        Difference between mean and median.
    MaxIntensity : float
        Max intensity data.
    MinIntensity : float
        Min intensity data.
    StdIntensity : float
        Standard deviation of intensity data.
    """

    pixOfInterest = I[Coords[:, 0], Coords[:, 1]]
    # compute intensity features
    MeanIntensity = np.mean(pixOfInterest)
    MeanMedianDifferenceIntensity = MeanIntensity - \
        np.median(pixOfInterest)
    MaxIntensity = max(pixOfInterest)
    MinIntensity = min(pixOfInterest)
    StdIntensity = np.std(pixOfInterest)

    return MeanIntensity, MeanMedianDifferenceIntensity, \
        MaxIntensity, MinIntensity, StdIntensity
