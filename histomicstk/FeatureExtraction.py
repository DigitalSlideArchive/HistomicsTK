from skimage.measure import regionprops
import numpy as np
import math
import pandas as pd


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
