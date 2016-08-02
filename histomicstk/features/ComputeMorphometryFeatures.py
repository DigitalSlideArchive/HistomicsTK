import numpy as np
import pandas as pd
from skimage.measure import regionprops


def ComputeMorphometryFeatures(im_label):
    """
    Calculates morphometry features for each object

    Parameters
    ----------
    im_label : array_like
        A labeled mask image wherein intensity of a pixel is the ID of the
        object it belongs to. Non-zero values are considered to be foreground
        objects.

    rprops[i] : property
        A rprops[i] property from regioprops.

    Returns
    -------
    fdata: pandas.DataFrame
        A pandas dataframe containing the morphometry features for each
        object/label listed below.

    Notes
    -----

    List of morphometry features computed by this function:

    Area : int
        Number of pixels the object occupies.

    Circularity: float
        A measure of how similar the shape of an object is to the circle

    Eccentricity : float
        A measure of aspect ratio computed to be the eccentricity of the
        ellipse that has the same second-moments as the region. Eccentricity
        of an ellipse is the ratio of the focal distance (distance between
        focal points) over the major axis length. The value is in the
        interval [0, 1). When it is 0, the ellipse becomes a circle.

    EquivalentDiameter : float
        The diameter of a circle with the same area as the object.

    Extent : float
        Ratio of area of the object to its axis-aligned bounding box.

    MajorAxisLength : float
        The length of the major axis of the ellipse that has the same
        normalized second central moments as the object.

    MinorAxisLength : float
        The length of the minor axis of the ellipse that has the same
        normalized second central moments as the region.

    MajorMinorAxisRatio : float

    Perimeter : float
        Perimeter of object which approximates the contour as a line
        through the centers of border pixels using a 4-connectivity.

    Solidity : float
        A measure of convexity computed as the ratio of the number of pixels
        in the object to that of its convex hull.
    """

    # List of feature names in alphabetical order
    feature_list = [
        'Area',
        'Circularity',
        'Eccentricity',
        'EquivalentDiameter'
        'Extent',
        'MajorAxisLength',
        'MinorAxisLength',
        'MajorMinorAxisRatio',
        'Perimeter',
        'Solidity',
    ]

    rprops = regionprops(im_label)

    numFeatures = len(feature_list)
    numLabels = len(rprops)
    fdata = pd.DataFrame(np.zeros((numLabels, numFeatures)),
                         columns=feature_list)

    for i in range(numLabels):

        # compute Area
        fdata.at[i, 'Area'] = rprops[i].area

        # compute Circularity
        numerator = 4 * np.pi * rprops[i].area
        denominator = rprops[i].perimeter**2
        if denominator:
            fdata.at[i, 'Circularity'] = numerator / denominator
        else:
            fdata.at[i, 'Circularity'] = 0  # should this be NaN?

        # compute Eccentricity
        fdata.at[i, 'Eccentricity'] = rprops[i].eccentricity

        # compute EquivalentDiameter
        fdata.at[i, 'EquivalentDiameter'] = rprops[i].equivalent_diameter

        # compute Extent
        fdata.at[i, 'Extent'] = rprops[i].extent

        # compute MajorAxisLength and MinorAxisLength
        fdata.at[i, 'MajorAxisLength'] = rprops[i].major_axis_length
        fdata.at[i, 'MinorAxisLength'] = rprops[i].minor_axis_length

        # compute MajorMinor axis ratios
        fdata.at[i, 'MajorMinorAxisRatio'] = \
            rprops[i].major_axis_length / rprops[i].minor_axis_length

        # compute Perimeter
        fdata.at[i, 'Perimeter'] = rprops[i].perimeter

        # compute Solidity
        fdata.at[i, 'Solidity'] = rprops[i].solidity

    return fdata
