from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
import numpy as np
import math
import pandas as pd
import collections


def FeatureExtraction(Label, I, K=128, Fs=6, Delta=8):
    """
    Calculate features from a label image.
    Parameters
    ----------
    Label : array_like
        A label image.
    I : array_like
        A intensity image.
    K : Number of points for boundary resampling to calculate fourier
        descriptors. Default value = 128.
    Fs : Number of frequency bins for calculating FSDs. Default value = 6.
    Delta : scalar, used to dilate nuclei and define cytoplasm region.
            Default value = 8.
    Returns
    -------
    df : 2-dimensional labeled data structure, float64
        Pandas data frame.
    See Also
    --------
    fft.fft : Compute the one-dimensional discrete Fourier Transform

    find_boundaries : Return bool array where boundaries between labeled
        regions are True.
    ##########     Features Start     ##########
    Slidename and centroids
    -----------------------
    Slide	X	Y
    Morphometry features
    --------------------
    Area	Perimeter	Eccentricity	Circularity
    MajorAxisLength MinorAxisLength	Extent Solidity
    FSDs
    ----
    FSD1	FSD2	FSD3	FSD4	FSD5	FSD6
    Hematoxlyin Features
    --------------------
    HematoxlyinMeanIntensity	HematoxlyinMeanMedianDifferenceIntensity
    HematoxlyinMaxIntensity	HematoxlyinMinIntensity	HematoxlyinStdIntensity
    HematoxlyinEntropy	HematoxlyinEnergy	HematoxlyinSkewness
    HematoxlyinKurtosis	HematoxlyinMeanGradMag	HematoxlyinStdGradMag
    HematoxlyinEntropyGradMag	HematoxlyinEnergyGradMag
    HematoxlyinSkewnessGradMag	HematoxlyinKurtosisGradMag
    HematoxlyinSumCanny	HematoxlyinMeanCanny
    Cytoplas Features
    --------------------
    CytoplasmMeanIntensity	CytoplasmMeanMedianDifferenceIntensity
    CytoplasmMaxIntensity	CytoplasmMinIntensity	CytoplasmStdIntensity
    CytoplasmEntropy	CytoplasmEnergy	CytoplasmSkewness	CytoplasmKurtosis
    CytoplasmMeanGradMag	CytoplasmStdGradMag	CytoplasmEntropyGradMag
    CytoplasmEnergyGradMag	CytoplasmSkewnessGradMag	CytoplasmKurtosisGradMag
    CytoplasmSumCanny	CytoplasmMeanCanny
    Boundary
    --------
    Boundaries: x1,y1;x2,y2;...
    ##########     Features End     ##########
    """

    # get total regions
    Num = Label.max()

    # get Label size x
    size_x = Label.shape[0]

    # initialize centroids
    CentroidX = []
    CentroidY = []

    # initialize morphometry features
    Area = []
    Perimeter = []
    Eccentricity = []
    Circularity = []
    MajorAxisLength = []
    MinorAxisLength = []
    Extent = []
    Solidity = []

    # initialize FSD feature group
    FSDGroup = []

    # initialize panda dataframe
    df = pd.DataFrame()

    # fourier descriptors, spaced evenly over the interval 1:K/2
    Interval = np.round(
        np.power(
            2, np.linspace(0, math.log(K, 2)-1, Fs+1, endpoint=True)
        )
    ).astype(np.uint8)

    # extract feature information
    for region in regionprops(Label, I):
        # add centroids
        CentroidX = np.append(CentroidX, region.centroid[0])
        CentroidY = np.append(CentroidY, region.centroid[1])
        # add morphometry features
        Area = np.append(Area, region.area)
        Perimeter = np.append(Perimeter, region.perimeter)
        Eccentricity = np.append(Eccentricity, region.eccentricity)
        Circularity = np.append(
            Circularity,
            4 * math.pi * region.area / math.pow(region.perimeter, 2)
        )
        MajorAxisLength = np.append(MajorAxisLength, region.major_axis_length)
        MinorAxisLength = np.append(MinorAxisLength, region.minor_axis_length)
        Extent = np.append(Extent, region.extent)
        Solidity = np.append(Solidity, region.solidity)
        # get bounds of dilated nucleus
        # region.bbox : min_row, min_col, max_row, max_col
        # bounds : min_row, max_row, min_col, max_col
        bounds = GetBounds(region.bbox, Delta, size_x)
        # grab nucleus mask
        Nucleus = (
            Label[bounds[0]:bounds[1], bounds[2]:bounds[3]] == region.label
        ).astype(np.int)
        # get Bounds
        Bounds = np.argwhere(
            find_boundaries(Nucleus, mode="inner").astype(np.uint8) == 1
        )
        # calculate FSDs
        FSD = FSDs(
            Bounds[:, 0], Bounds[:, 1],
            K, Interval
        )
        FSDGroup = np.append(FSDGroup, FSD)

    # add columns to dataframe
    df['X'] = CentroidX
    df['Y'] = CentroidY

    df['Area'] = Area
    df['Perimeter'] = Perimeter
    df['Eccentricity'] = Eccentricity
    df['Circularity'] = Circularity
    df['MajorAxisLength'] = MajorAxisLength
    df['MinorAxisLength'] = MinorAxisLength
    df['Extent'] = Extent
    df['Solidity'] = Solidity

    FSDGroup = FSDGroup.reshape(Num, Fs)

    df['FSD1'] = FSDGroup[:, 0]
    df['FSD2'] = FSDGroup[:, 1]
    df['FSD3'] = FSDGroup[:, 2]
    df['FSD4'] = FSDGroup[:, 3]
    df['FSD5'] = FSDGroup[:, 4]
    df['FSD6'] = FSDGroup[:, 5]

    return df


def GetBounds(bbox, delta, N):
    """
    Returns bounds of object in global label image.
    """
    bounds = np.zeros(4, dtype=np.int8)
    bounds[0] = max(0, math.floor(bbox[0] - delta))
    bounds[1] = min(N-1, math.ceil(bbox[0] + bbox[2] + delta))
    bounds[2] = max(0, math.floor(bbox[1] - delta))
    bounds[3] = min(N-1, math.ceil(bbox[1] + bbox[3] + delta))

    return bounds


def InterpolateArcLength(X, Y, L):
    """
    Resamples boundary points [X, Y] at L total equal arc-length locations.

    Returns
    -------
    iX - L-length vector of horizontal interpolated coordinates with equal
         arc-length spacing.
    iY - L-length vector of vertical interpolated coordinates with equal
         arc-length spacing.
    """

    # length of X
    K = len(X)

    # generate spaced points
    Interval = np.linspace(0, 1, L)

    # get segment lengths
    Lengths = np.sqrt(
        np.power(np.diff(X), 2) + np.power(np.diff(Y), 2)
    )

    # normalize to unit length
    Lengths = Lengths / Lengths.sum()

    # calculate cumulative length along boundary
    Cumulative = np.hstack((0., np.cumsum(Lengths)))

    # place points in 'Interval' along boundary
    Locations = np.digitize(Interval, Cumulative)

    # clip to ends
    Locations[Locations < 1] = 1
    Locations[Locations >= K] = K - 1
    Locations = Locations - 1

    # linear interpolation
    Lie = np.divide(
        (Interval - [Cumulative[i] for i in Locations]),
        [Lengths[i] for i in Locations]
    )

    tX = np.array([X[i] for i in Locations])
    tY = np.array([Y[i] for i in Locations])
    iX = tX + np.multiply(
        np.array([X[i+1] for i in Locations]) - tX, Lie
    )
    iY = tY + np.multiply(
        np.array([Y[i+1] for i in Locations]) - tY, Lie
    )

    iXY = collections.namedtuple('iXY', ['iX', 'iY'])
    Output = iXY(iX, iY)

    return Output


def FSDs(X, Y, K, Intervals):
    """
    Calculated FSDs from boundary points X,Y. Boundaries are resampled to have
    K equally spaced points (arclength) around the shape. The curvature is
    calculated using the cumulative angular function, measuring the
    displacement of the tangent angle from the starting point of the boundary.
    The K-length fft of the cumulative angular function is calculated, and
    then the elements of 'F' are summed as the spectral energy over
    'Intervals'.

    Returns F - length(Intervals) vector containing spectral energy of
    cumulative angular function, summed over defined 'Intervals'.

    References
    ----------
    .. [1] D. Zhang et al. "A comparative study on shape retrieval using
    Fourier descriptors with different shape signatures," In Proc.
    ICIMADE01, 2001.
    """

    # check input 'Intervals'
    if Intervals[0] != 1.:
        Intervals = np.hstack((1., Intervals))
    if Intervals[-1] != (K / 2):
        Intervals = np.hstack((Intervals, float(K)))

    # get length of intervals
    L = len(Intervals)

    # interpolate boundaries
    iXY = InterpolateArcLength(X, Y, K)

    # calculate curvature
    Curvature = np.arctan2(
        (iXY.iY[1:] - iXY.iY[:-1]),
        (iXY.iX[1:] - iXY.iX[:-1])
    )

    # make curvature cumulative
    Curvature = Curvature - Curvature[0]

    # calculate FFT
    fX = np.fft.fft(Curvature).T

    # spectral energy
    fX = fX * fX.conj()
    fX = fX / fX.sum()

    # calculate 'F' values
    F = []

    for i in range(L-1):
        f = np.round(
            fX[Intervals[i]-1:Intervals[i+1]].sum(), L
        )
        F = np.append(F, f).real.astype(float)

    return F
