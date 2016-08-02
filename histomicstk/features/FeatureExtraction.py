from .ComputeFSDs import ComputeFSDs
from .ComputeGradientFeatures import ComputeGradientFeatures
from .ComputeIntensityFeatures import ComputeIntensityFeatures
from .ComputeMorphometryFeatures import ComputeMorphometryFeatures
import numpy as np
import pandas as pd
from skimage.feature import canny
from skimage.measure import regionprops
from skimage.morphology import disk, binary_dilation


def FeatureExtraction(Label, In, Ic, K=128, Fs=6, Delta=8):
    """
    Calculates features from a label image.

    Parameters
    ----------
    Label : array_like
        A M x N label image.
    In : array_like
        A M x N intensity image for Nuclei.
    Ic : array_like
        A M x N intensity image for Cytoplasms.
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

    - `Morphometry features`:
        - CentroidsX,
        - CentroidsY,
        - Area,
        - Perimeter,
        - MajorAxisLength,
        - MinorAxisLength,
        - MajorMinorAxisRatio,
        - MajorAxisCoordsX,
        - MajorAxisCoordsY,
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

    # get Label size x
    size_x = Label.shape[0]
    size_y = Label.shape[1]

    # get the number of objects in Label
    regions = regionprops(Label)
    num = len(regions)

    # initialize morphometry feature group
    MorphometryGroup = np.zeros((num, 13))

    # initialize FSD feature group
    FSDGroup = np.zeros((num, Fs))

    # initialize gradient feature groups
    HematoxylinGradientGroup = np.zeros((num, 8))
    EosinGradientGroup = np.zeros((num, 8))

    # initialize intensity feature groups
    HematoxylinIntensityGroup = np.zeros((num, 9))
    EosinIntensityGroup = np.zeros((num, 9))

    # create round structuring element
    Disk = disk(Delta)

    # compute bw canny and gradient difference for H and E
    Gnx, Gny = np.gradient(In)
    diffGn = np.sqrt(Gnx**2 + Gny**2)
    BW_cannyn = canny(In)

    Gcx, Gcy = np.gradient(Ic)
    diffGc = np.sqrt(Gcx**2 + Gcy**2)
    BW_cannyc = canny(Ic)

    # do feature extraction
    for i in range(0, num):

        # get bounds of dilated nucleus
        min_row, max_row, min_col, max_col = \
            GetBounds(regions[i].bbox, Delta, size_x, size_y)
        # grab nucleus mask
        Nucleus = (
            Label[min_row:max_row, min_col:max_col] == regions[i].label
        ).astype(np.bool)
        # compute Fourier shape descriptors
        FSDGroup[i, :] = ComputeFSDs(Nucleus, K, Fs)
        # generate object coords for nuclei and cytoplasmic regions
        Nuclei = regions[i].coords
        # compute Texture, Gradient, Intensity features
        HematoxylinGradientGroup[i, :] = \
            ComputeGradientFeatures(In, Nuclei, diffGn, BW_cannyn)
        HematoxylinIntensityGroup[i, :] = \
            ComputeIntensityFeatures(In, Nuclei)
        # get mask for all nuclei in neighborhood
        Mask = (
            Label[min_row:max_row, min_col:max_col] > 0
        ).astype(np.uint8)
        # remove nucleus region from cytoplasm+nucleus mask
        cytoplasm = (
            np.logical_xor(Mask, binary_dilation(Nucleus, Disk))
        )
        # get list of cytoplasm pixels
        regionCoords = np.argwhere(cytoplasm == 1)
        regionCoords[:, 0] = regionCoords[:, 0] + min_row
        regionCoords[:, 1] = regionCoords[:, 1] + min_col
        # compute Texture, Gradient, Intensity features
        EosinGradientGroup[i, :] = \
            ComputeGradientFeatures(Ic, regionCoords, diffGc, BW_cannyc)
        EosinIntensityGroup[i, :] = \
            ComputeIntensityFeatures(Ic, regionCoords)

    # initialize panda dataframe
    df = pd.DataFrame()

    fmorph = ComputeMorphometryFeatures(Label)
    df = pd.concat([df, fmorph], axis=1)

    for i in range(0, Fs):
        df['FSD' + str(i+1)] = FSDGroup[:, i]

    GradientNames = ['MeanGradMag', 'StdGradMag', 'EntropyGradMag',
                     'EnergyGradMag', 'SkewnessGradMag', 'KurtosisGradMag',
                     'SumCanny', 'MeanCanny']

    for i in range(0, len(GradientNames)):
        df['Hematoxylin' + GradientNames[i]] = \
            HematoxylinGradientGroup[:, i]
        df['Cytoplasm' + GradientNames[i]] = EosinGradientGroup[:, i]

    IntensityNames = ['MeanIntensity', 'MeanMedianDifferenceIntensity',
                      'MaxIntensity', 'MinIntensity', 'StdIntensity',
                      'Entropy', 'Energy', 'Skewness', 'Kurtosis']

    for i in range(0, len(IntensityNames)):
        df['Hematoxylin' + IntensityNames[i]] = \
            HematoxylinIntensityGroup[:, i]
        df['Cytoplasm' + IntensityNames[i]] = EosinIntensityGroup[:, i]

    return df


def GetBounds(bbox, delta, M, N):
    """
    Returns bounds of object in global label image.

    Parameters
    ----------
    bbox : tuple
        Bounding box (min_row, min_col, max_row, max_col).
    delta : int
        Used to dilate nuclei and define cytoplasm region.
        Default value = 8.
    M : int
        X size of label image.
    N : int
        Y size of label image.

    Returns
    -------
    min_row : int
        Minum row of the region bounds.
    max_row : int
        Maximum row of the region bounds.
    min_col : int
        Minum column of the region bounds.
    max_col : int
        Maximum column of the region bounds.
    """

    min_row, min_col, max_row, max_col = bbox

    min_row_out = max(0, (min_row - delta))
    max_row_out = min(M-1, (max_row + delta))
    min_col_out = max(0, (min_col - delta))
    max_col_out = min(N-1, (max_col + delta))

    return min_row_out, max_row_out, min_col_out, max_col_out
