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
    fdata: Pandas data frame containing the following morphometry features
    for each object/label
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
    """

    # feature names listed in alphabetical order
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
    fdata = pd.DataFrame(np.zeros(numLabels, numFeatures),
                         columns=feature_list)

    for i in range(numLabels):

        # compute Area
        fdata.at[i, 'Area'] = rprops[i].area

        # compute Circularity
        numerator = 4 * np.pi * rprops[i].Area
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
