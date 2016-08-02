import numpy as np


def ComputeMorphometryFeatures(Region):
    """
    Calculates morphometry features from a region property

    Parameters
    ----------
    Region : property
        A region property from regioprops.

    Returns
    -------
    MGroup : array_like
        A 1 x 13 morphometry features.

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
    """

    MGroup = np.zeros(13)
    # compute Centroids
    MGroup[0] = Region.centroid[0]
    MGroup[1] = Region.centroid[1]
    # compute Area
    MGroup[2] = Region.area
    # compute Perimeter
    MGroup[3] = Region.perimeter
    # compute Eccentricity
    MGroup[4] = Region.eccentricity
    # compute Circularity
    numerator = 4 * np.pi * MGroup[2]
    denominator = np.power(MGroup[3], 2)
    MGroup[5] = numerator / denominator if denominator else 0
    # compute MajorAxisLength and MinorAxisLength
    MGroup[6] = Region.major_axis_length
    MGroup[7] = Region.minor_axis_length
    # compute MajorMinor axis ratios
    MGroup[8] = MGroup[6]/MGroup[7]
    # get region orientation
    ot = Region.orientation
    # find length of Maxjor X and Y
    if ot < 0:
        lengthofMajorX = (MGroup[6]/2) * np.sin(ot)
        lengthofMajorY = (MGroup[6]/2) * np.cos(ot) * (-1)
    else:
        lengthofMajorX = (MGroup[6]/2) * np.sin(ot) * (-1)
        lengthofMajorY = (MGroup[6]/2) * np.cos(ot)
    # add lengths to Centroids
    MajorAxisX = MGroup[0] + lengthofMajorX
    MajorAxisY = MGroup[1] + lengthofMajorY
    # get MajorAxisCoords
    MGroup[9] = np.ceil(MajorAxisX)
    MGroup[10] = np.ceil(MajorAxisY)
    # compute Extent
    MGroup[11] = Region.extent
    # compute Solidity
    MGroup[12] = Region.solidity

    return MGroup
