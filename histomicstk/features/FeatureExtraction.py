import collections
import math
import numpy as np
import pandas as pd
from scipy.stats import entropy, skew, kurtosis
from skimage.feature import canny
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from skimage.morphology import disk, dilation


def FeatureExtraction(Label, In, Ic, W, K=128, Fs=6, Delta=8):
    """
    Calculates features from a label image.

    Parameters
    ----------
    Label : array_like
        A T x T label image.
    In : array_like
        A T x T intensity image for Nuclei.
    Ic : array_like
        A T x T intensity image for Cytoplasms.
    W : array_like
        A 3x3 matrix containing the stain colors in its columns.
        In the case of two stains, the third column is zero and will be
        complemented using cross-product. The matrix should contain a
        minumum two nonzero columns.
    K : Number of points for boundary resampling to calculate fourier
        descriptors. Default value = 128.
    Fs : Number of frequency bins for calculating FSDs. Default value = 6.
    Delta : scalar, used to dilate nuclei and define cytoplasm region.
            Default value = 8.
    Returns
    -------
    df : 2-dimensional labeled data structure, float64
        Pandas data frame.
    Notes
    -----
    The following features are computed:

    - `Centroids`:
        - X,Y

    - `Morphometry features`:
        - Area,
        - Perimeter,
        - MajorAxisLength,
        - MinorAxisLength,
        - Eccentricity,
        - Circularity,
        - Extent,
        - Solidity

    - `Fourier shape descriptors`:
        - FSD1-FSD6

    - Intensity features for hematoxylin and cytoplasm channels:
        - MinIntensity, MaxIntensity,
        - MeanIntensity, StdIntensity,
        - MeanMedianDifferenceIntensity,
        - Entropy, Energy, Skewness and Kurtosis

    - Gradient/edge features for hematoxylin and cytoplasm channels:
        - MeanGradMag, StdGradMag, SkewnessGradMag, KurtosisGradMag,
        - EntropyGradMag, EnergyGradMag,
        - SumCanny, MeanCanny
    References
    ----------
    .. [1] D. Zhang et al. "A comparative study on shape retrieval using
       Fourier descriptors with different shape signatures," In Proc.
       ICIMADE01, 2001.
    .. [2] Daniel Zwillinger and Stephen Kokoska. "CRC standard probability
       and statistics tables and formulae," Crc Press, 1999.
    """

    # get total regions
    NumofLabels = Label.max()

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
    FSDGroup = np.zeros((NumofLabels, Fs))

    # initialize Nuclei, Cytoplasms
    Nuclei = [[] for i in range(NumofLabels)]
    Cytoplasms = [[] for i in range(NumofLabels)]

    # create round structuring element
    Disk = disk(Delta)

    # initialize panda dataframe
    df = pd.DataFrame()

    # fourier descriptors, spaced evenly over the interval 1:K/2
    Interval = np.round(
        np.power(
            2, np.linspace(0, math.log(K, 2)-1, Fs+1, endpoint=True)
        )
    ).astype(np.uint8)

    # extract feature information
    for region in regionprops(Label):
        # add centroids
        CentroidX = np.append(CentroidX, region.centroid[0])
        CentroidY = np.append(CentroidY, region.centroid[1])
        # add morphometry features
        Area = np.append(Area, region.area)
        Perimeter = np.append(Perimeter, region.perimeter)
        Eccentricity = np.append(Eccentricity, region.eccentricity)
        numerator = 4 * math.pi * region.area
        denominator = math.pow(region.perimeter, 2)
        Circularity = np.append(
            Circularity,
            numerator / denominator if denominator else 0
        )
        MajorAxisLength = np.append(MajorAxisLength, region.major_axis_length)
        MinorAxisLength = np.append(MinorAxisLength, region.minor_axis_length)
        Extent = np.append(Extent, region.extent)
        Solidity = np.append(Solidity, region.solidity)
        # get bounds of dilated nucleus
        bounds = GetBounds(region.bbox, Delta, size_x)
        # grab nucleus mask
        Nucleus = (
            Label[bounds[0]:bounds[1], bounds[2]:bounds[3]] == region.label
        ).astype(np.uint8)
        # find nucleus boundaries
        Bounds = np.argwhere(
            find_boundaries(Nucleus, mode="inner").astype(np.uint8) == 1
        )
        # calculate and add FSDs
        FSDGroup[region.label-1, :] = FSDs(
            Bounds[:, 0], Bounds[:, 1],
            K, Interval
        )
        # generate object coords for nuclei and cytoplasmic regions
        Nuclei[region.label-1] = region.coords
        # get mask for all nuclei in neighborhood
        Mask = (
            Label[bounds[0]:bounds[1], bounds[2]:bounds[3]] > 0
        ).astype(np.uint8)
        # remove nucleus region from cytoplasm+nucleus mask
        cytoplasm = (
            np.logical_xor(Mask, dilation(Nucleus, Disk))
        ).astype(np.uint8)
        # get list of cytoplasm pixels
        Cytoplasms[region.label-1] = GetPixCoords(cytoplasm, bounds)

    # calculate hematoxlyin features, capture feature names
    HematoxylinIntensityGroup = IntensityFeatureGroup(In, Nuclei)
    HematoxylinTextureGroup = TextureFeatureGroup(In, Nuclei)
    HematoxylinGradientGroup = GradientFeatureGroup(In, Nuclei)
    # calculate eosin features
    EosinIntensityGroup = IntensityFeatureGroup(Ic, Cytoplasms)
    EosinTextureGroup = TextureFeatureGroup(Ic, Cytoplasms)
    EosinGradientGroup = GradientFeatureGroup(Ic, Cytoplasms)

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

    for i in range(0, Fs):
        df['FSD' + str(i+1)] = FSDGroup[:, i]

    for f in HematoxylinIntensityGroup._fields:
        df['Hematoxylin' + f] = getattr(HematoxylinIntensityGroup, f)

    for f in HematoxylinTextureGroup._fields:
        df['Hematoxylin' + f] = getattr(HematoxylinTextureGroup, f)

    for f in HematoxylinGradientGroup._fields:
        df['Hematoxylin' + f] = getattr(HematoxylinGradientGroup, f)

    for f in EosinIntensityGroup._fields:
        df['Cytoplasm' + f] = getattr(EosinIntensityGroup, f)

    for f in EosinTextureGroup._fields:
        df['Cytoplasm' + f] = getattr(EosinTextureGroup, f)

    for f in EosinGradientGroup._fields:
        df['Cytoplasm' + f] = getattr(EosinGradientGroup, f)

    return df


def GradientFeatureGroup(I, Coords):
    """
    Get GradientFeatures for nuclei and cytoplasms
    Parameters
    ----------
    I : array_like
        A T x T intensity image.
    Coords : array_like
        A N x 2 coordinate list of a region.
    Returns
    -------
    MeanGradMag : array_like
        Mean of gradient data.
    StdGradMag : array_like
        Standard deviation of gradient data.
    EntropyGradMag : array_like
        Entroy of gradient data.
    EnergyGradMag : array_like
        Energy of gradient data.
    SkewnessGradMag : array_like
        Skewness of gradient data. Value is 0 when all values are equal.
    KurtosisGradMag : array_like
        Kurtosis of gradient data. Value is -3 when all values are equal.
    SumCanny : array_like
        Sum of canny filtered gradient data.
    MeanCanny : array_like
        Mean of canny filtered gradient data.
    Notes
    -----
    Return values are returned as a namedtuple.
    References
    ----------
    .. [1] Daniel Zwillinger and Stephen Kokoska. "CRC standard probability
    and statistics tables and formulae," Crc Press, 1999.
    """
    Gx, Gy = np.gradient(I)
    diffG = np.sqrt(Gx*Gx + Gy*Gy)
    BW_canny = canny(I)

    f = np.zeros((len(Coords), 8))
    for i in range(len(Coords)):
        if len(Coords[i]) != 0:
            pixOfInterest = diffG[Coords[i][:, 0], Coords[i][:, 1]]
            f[i, 0] = np.mean(pixOfInterest)
            f[i, 1] = np.std(pixOfInterest)
            f[i, 2] = entropy(pixOfInterest)
            hist, bins = np.histogram(pixOfInterest, bins=np.arange(256))
            prob = hist/np.sum(hist, dtype=np.float32)
            f[i, 3] = np.sum(np.power(prob, 2))
            f[i, 4] = skew(pixOfInterest)
            f[i, 5] = kurtosis(pixOfInterest)
            bw_canny = BW_canny[Coords[i][:, 0], Coords[i][:, 1]]
            f[i, 6] = np.sum(bw_canny)
            f[i, 7] = f[i, 6] / len(pixOfInterest)

    MeanGradMag = f[:, 0]
    StdGradMag = f[:, 1]
    EntropyGradMag = f[:, 2]
    EnergyGradMag = f[:, 3]
    SkewnessGradMag = f[:, 4]
    KurtosisGradMag = f[:, 5]
    SumCanny = f[:, 6]
    MeanCanny = f[:, 7]

    iFG = collections.namedtuple(
        'iFG',
        [
            'MeanGradMag',
            'StdGradMag',
            'EntropyGradMag',
            'EnergyGradMag',
            'SkewnessGradMag',
            'KurtosisGradMag',
            'SumCanny',
            'MeanCanny'
        ]
    )
    Output = iFG(
        MeanGradMag, StdGradMag, EntropyGradMag, EnergyGradMag,
        SkewnessGradMag, KurtosisGradMag, SumCanny, MeanCanny
    )

    return Output


def TextureFeatureGroup(I, Coords):
    """
    Get TextureFeatures for nuclei and cytoplasms
    Parameters
    ----------
    I : array_like
        A T x T intensity image.
    Coords : array_like
        A N x 2 coordinate list of a region.
    Returns
    -------
    Entropy : array_like
        Entroy of intensity data.
    Energy : array_like
        Energy of intensity data.
    Skewness : array_like
        Skewness of intensity data. Value is 0 when all values are equal.
    Kurtosis : array_like
        Kurtosis of intensity data. Value is -3 when all values are equal.
    Notes
    -----
    Return values are returned as a namedtuple.
    References
    ----------
    .. [1] Daniel Zwillinger and Stephen Kokoska. "CRC standard probability
       and statistics tables and formulae," Crc Press, 1999.
    """
    f = np.zeros((len(Coords), 4))
    for i in range(len(Coords)):
        if len(Coords[i]) != 0:
            pixOfInterest = I[Coords[i][:, 0], Coords[i][:, 1]]
            hist, bins = np.histogram(pixOfInterest, bins=np.arange(256))
            prob = hist/np.sum(hist, dtype=np.float32)
            f[i, 0] = entropy(pixOfInterest)
            f[i, 1] = np.sum(np.power(prob, 2))
            f[i, 2] = skew(pixOfInterest)
            f[i, 3] = kurtosis(pixOfInterest)

    Entropy = f[:, 0]
    Energy = f[:, 1]
    Skewness = f[:, 2]
    Kurtosis = f[:, 3]

    iFG = collections.namedtuple(
        'iFG',
        [
            'Entropy',
            'Energy',
            'Skewness',
            'Kurtosis'
        ]
    )
    Output = iFG(
        Entropy, Energy, Skewness, Kurtosis
    )

    return Output


def IntensityFeatureGroup(I, Coords):
    """
    Get IntensityFeatures for nuclei and cytoplasms
    Parameters
    ----------
    I : array_like
        A T x T intensity image.
    Coords : array_like
        A N x 2 coordinate list of a region.
    Returns
    -------
    MeanIntensity : array_like
        Mean of intensity data.
    MeanMedianDifferenceIntensity : array_like
        Difference between mean and median.
    MaxIntensity : array_like
        Max intensity data.
    MinIntensity : array_like
        Min intensity data.
    StdIntensity : array_like
        Standard deviation of intensity data.
    Notes
    -----
    Return values are returned as a namedtuple.
    """
    f = np.zeros((len(Coords), 5))
    for i in range(len(Coords)):
        if len(Coords[i]) != 0:
            pixOfInterest = I[Coords[i][:, 0], Coords[i][:, 1]]
            f[i, 0] = np.mean(pixOfInterest)
            f[i, 1] = f[i, 0] - np.median(pixOfInterest)
            f[i, 2] = max(pixOfInterest)
            f[i, 3] = min(pixOfInterest)
            f[i, 4] = np.std(pixOfInterest)

    MeanIntensity = f[:, 0]
    MeanMedianDifferenceIntensity = f[:, 1]
    MaxIntensity = f[:, 2]
    MinIntensity = f[:, 3]
    StdIntensity = f[:, 4]

    iFG = collections.namedtuple(
        'iFG',
        [
            'MeanIntensity',
            'MeanMedianDifferenceIntensity',
            'MaxIntensity',
            'MinIntensity',
            'StdIntensity'
        ]
    )
    Output = iFG(
        MeanIntensity, MeanMedianDifferenceIntensity,
        MaxIntensity, MinIntensity, StdIntensity
    )

    return Output


def GetPixCoords(Binary, bounds):
    """
    Get global coords of object extracted from tile.
    Parameters
    ----------
    Binary : array_like
        A binary image.
    bounds : array_like
        A region bounds. [min_row, max_row, min_col, max_col].
    Returns
    -------
    coords : array_like
        A N x 2 list of coordinate for a region.
    """
    coords = np.where(Binary == 1)
    coords = np.asarray(coords)
    coords[0] = np.add(coords[0], bounds[0])
    coords[1] = np.add(coords[1], bounds[2])
    coords = coords.T

    return coords


def GetBounds(bbox, delta, N):
    """
    Returns bounds of object in global label image.
    Parameters
    ----------
    bbox : tuple
        Bounding box (min_row, min_col, max_row, max_col).
    delta : int
        Used to dilate nuclei and define cytoplasm region.
        Default value = 8.
    N : int
        X or Y Size of label image.
    Returns
    -------
    bounds : array_like
        A region bounds. [min_row, max_row, min_col, max_col].
    """
    bounds = np.zeros(4, dtype=np.uint8)
    bounds[0] = max(0, math.floor(bbox[0] - delta))
    bounds[1] = min(N-1, math.ceil(bbox[0] + bbox[2] + delta))
    bounds[2] = max(0, math.floor(bbox[1] - delta))
    bounds[3] = min(N-1, math.ceil(bbox[1] + bbox[3] + delta))

    return bounds


def InterpolateArcLength(X, Y, L):
    """
    Resamples boundary points [X, Y] at L total equal arc-length locations.
    Parameters
    ----------
    X : array_like
        x points of boundaries
    Y : array_like
        y points of boundaries
    L : int
        Number of points for boundary resampling to calculate fourier
        descriptors. Default value = 128.
    Returns
    -------
    iX : array_like
        L-length vector of horizontal interpolated coordinates with equal
        arc-length spacing.
    iY : array_like
        L-length vector of vertical interpolated coordinates with equal
        arc-length spacing.
    Notes
    -----
    Return values are returned as a namedtuple.
    """

    # length of X
    K = len(X)
    # initialize iX, iY
    iX = np.zeros((0,))
    iY = np.zeros((0,))
    # generate spaced points
    Interval = np.linspace(0, 1, L)
    # get segment lengths
    Lengths = np.sqrt(
        np.power(np.diff(X), 2) + np.power(np.diff(Y), 2)
    )
    # check Lengths
    if Lengths.size:
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
    Parameters
    ----------
    X : array_like
        x points of boundaries
    Y : array_like
        y points of boundaries
    K : int
        Number of points for boundary resampling to calculate fourier
        descriptors. Default value = 128.
    Intervals : array_like
        Intervals spaced evenly over 1:K/2.
    Returns
    -------
    F : array_like
        length(Intervals) vector containing spectral energy of
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
    # initialize F
    F = np.zeros((L-1, ))
    # interpolate boundaries
    iXY = InterpolateArcLength(X, Y, K)
    # check if iXY.iX is not empty
    if iXY.iX.size:
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
        for i in range(L-1):
            F[i] = np.round(
                fX[Intervals[i]-1:Intervals[i+1]].sum(), L
            ).real.astype(float)

    return F
