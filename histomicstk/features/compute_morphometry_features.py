import numpy as np
import pandas as pd
from skimage.measure import regionprops


def compute_morphometry_features(im_label, rprops=None):
    """
    Calculates morphometry features for each object

    Parameters
    ----------
    im_label : array_like
        A labeled mask image wherein intensity of a pixel is the ID of the
        object it belongs to. Non-zero values are considered to be foreground
        objects.

    rprops : output of skimage.measure.regionprops, optional
        rprops = skimage.measure.regionprops( im_label ). If rprops is not
        passed then it will be computed inside which will increase the
        computation time.

    Returns
    -------
    fdata: pandas.DataFrame
        A pandas dataframe containing the morphometry features for each
        object/label listed below.

    Notes
    -----
    List of morphometry features computed by this function:

    Size.Area : int
        Number of pixels the object occupies.

    Size.MajorAxisLength : float
        The length of the major axis of the ellipse that has the same
        normalized second central moments as the object.

    Size.MinorAxisLength : float
        The length of the minor axis of the ellipse that has the same
        normalized second central moments as the region.

    Size.Perimeter : float
        Perimeter of object which approximates the contour as a line
        through the centers of border pixels using a 4-connectivity.

    Shape.Circularity: float
        A measure of how similar the shape of an object is to the circle

    Shape.Eccentricity : float
        A measure of aspect ratio computed to be the eccentricity of the
        ellipse that has the same second-moments as the object region.
        Eccentricity of an ellipse is the ratio of the focal distance
        (distance between focal points) over the major axis length. The value
        is in the interval [0, 1). When it is 0, the ellipse becomes a circle.

    Shape.EquivalentDiameter : float
        The diameter of a circle with the same area as the object.

    Shape.Extent : float
        Ratio of area of the object to its axis-aligned bounding box.

    Shape.MinorMajorAxisRatio : float
        A measure of aspect ratio. Ratio of minor to major axis of the ellipse
        that has the same second-moments as the object region

    Shape.Solidity : float
        A measure of convexity computed as the ratio of the number of pixels
        in the object to that of its convex hull.

    """

    # List of feature names in alphabetical order
    feature_list = [
        'Size.Area',
        'Size.MajorAxisLength',
        'Size.MinorAxisLength',
        'Size.Perimeter',
        'Shape.Circularity',
        'Shape.Eccentricity',
        'Shape.EquivalentDiameter',
        'Shape.Extent',
        'Shape.MinorMajorAxisRatio',
        'Shape.Solidity',
    ]

    # compute object properties if not provided
    if rprops is None:
        rprops = regionprops(im_label)

    # create pandas data frame containing the features for each object
    numFeatures = len(feature_list)
    numLabels = len(rprops)
    fdata = pd.DataFrame(np.zeros((numLabels, numFeatures)),
                         columns=feature_list)

    for i in range(numLabels):

        # compute Area
        fdata.at[i, 'Size.Area'] = rprops[i].area

        # compute MajorAxisLength and MinorAxisLength
        # A bug in scikit-image could produce a (very slightly)
        # negative element in inertia_tensor_eigvals, so insert
        # "max(0, )" call here before invoking np.sqrt().
        inertia_tensor_eigvals = rprops[i].inertia_tensor_eigvals
        major_axis_length = 4 * np.sqrt(
            max(inertia_tensor_eigvals[0], 0))
        minor_axis_length = 4 * np.sqrt(
            max(inertia_tensor_eigvals[-1], 0))

        fdata.at[i, 'Size.MajorAxisLength'] = major_axis_length
        fdata.at[i, 'Size.MinorAxisLength'] = minor_axis_length

        # compute Perimeter
        fdata.at[i, 'Size.Perimeter'] = rprops[i].perimeter

        # compute Circularity
        numerator = 4 * np.pi * rprops[i].area
        denominator = rprops[i].perimeter**2
        if denominator:
            fdata.at[i, 'Shape.Circularity'] = numerator / denominator
        else:
            fdata.at[i, 'Shape.Circularity'] = 0  # should this be NaN?

        # compute Eccentricity
        fdata.at[i, 'Shape.Eccentricity'] = rprops[i].eccentricity

        # compute EquivalentDiameter
        fdata.at[i, 'Shape.EquivalentDiameter'] = rprops[i].equivalent_diameter

        # compute Extent
        fdata.at[i, 'Shape.Extent'] = rprops[i].extent

        # compute Minor to Major axis ratio
        if major_axis_length > 0:
            fdata.at[i, 'Shape.MinorMajorAxisRatio'] = \
                minor_axis_length / major_axis_length
        else:
            fdata.at[i, 'Shape.MinorMajorAxisRatio'] = 1

        # compute Solidity
        fdata.at[i, 'Shape.Solidity'] = rprops[i].solidity

    return fdata
