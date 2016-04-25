from skimage.measure import regionprops
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
    df : Pandas data frame.
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

    # initialize panda dataframe
    df = pd.DataFrame()

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


def log2space(a, b, n):
    """
    Returns n logarithmically-spaced points from 2^a to 2^b.
    """

    A = np.power(
        2, [a + x*(b-a)/(n-1) for x in range(0, n-1)]
    )

    B = np.power(
        2, b
    )

    A = np.hstack((A, B))

    return A
