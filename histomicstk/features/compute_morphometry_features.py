from collections import OrderedDict

import numpy as np


def compute_morphometry_features(im_label, rprops=None):
    """
    Calculate morphometry features for each object

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

    Orientation.Orientation :  float
        Angle between the horizontal axis and the major axis of the ellipse
        that has the same second moments as the region,
        ranging from `-pi/2` to `pi/2` counter-clockwise.

    Size.Area : int
        Number of pixels the object occupies.

    Size.ConvexHullArea :  int
        Number of pixels of convex hull image, which is the smallest convex
        polygon that encloses the region.

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

    Shape.FractalDimension : float
        Minkowski–Bouligand dimension, aka. the box-counting dimension. It
        is a measure of boundary complexity. See
        https://en.wikipedia.org/wiki/Minkowski%E2%80%93Bouligand_dimension

    Shape.MinorMajorAxisRatio : float
        A measure of aspect ratio. Ratio of minor to major axis of the ellipse
        that has the same second-moments as the object region

    Shape.Solidity : float
        A measure of convexity computed as the ratio of the number of pixels
        in the object to that of its convex hull.

    Shape.HuMoments-k : float
        Where k ranges from 1-7 are the 7 Hu moments features. The first six
        moments are translation, scale and rotation invariant, while the
        seventh moment flips its sign if the shape is a mirror image.
        See https://learnopencv.com/shape-matching-using-hu-moments-c-python/

    Shape.WeightedHuMoments-k : float
        Same as Hu moments, but instead of using the binary mask, using the
        intensity image.

    """
    import pandas as pd
    from skimage.measure import regionprops

    # compute object properties if not provided
    if rprops is None:
        rprops = regionprops(im_label)
    intensity_wtd = rprops[0]._intensity_image is not None

    # List of feature names
    featname_map = OrderedDict({
        'Orientation.Orientation': 'orientation',
        'Size.Area': 'area',
        'Size.ConvexHullArea': 'convex_area',
        'Size.MajorAxisLength': 'major_axis_length',
        'Size.MinorAxisLength': 'minor_axis_length',
        'Size.Perimeter': 'perimeter',
        'Shape.Circularity': None,
        'Shape.Eccentricity': 'eccentricity',
        'Shape.EquivalentDiameter': 'equivalent_diameter',
        'Shape.Extent': 'extent',
        'Shape.FractalDimension': None,
        'Shape.MinorMajorAxisRatio': None,
        'Shape.Solidity': 'solidity',
    })
    hu_cols = ['Shape.HuMoments%d' % k for k in range(1, 8)]
    featname_map.update({col: None for col in hu_cols})
    if intensity_wtd:
        wtd_hu_cols = [col.replace('.Hu', '.WeightedHu') for col in hu_cols]
        featname_map.update({col: None for col in wtd_hu_cols})
    feature_list = featname_map.keys()
    mapped_feats = [k for k, v in featname_map.items() if v is not None]

    # create pandas data frame containing the features for each object
    numFeatures = len(feature_list)
    numLabels = len(rprops)
    fdata = pd.DataFrame(np.zeros((numLabels, numFeatures)),
                         columns=feature_list)

    for i, nprop in enumerate(rprops):

        # assign some features from sklearn as-is
        for name in mapped_feats:
            fdata.at[i, name] = nprop[featname_map[name]]

        # compute Circularity
        numerator = 4 * np.pi * nprop.area
        denominator = nprop.perimeter ** 2
        if denominator > 0:
            fdata.at[i, 'Shape.Circularity'] = numerator / denominator

        # compute fractal dimension (boundary complexity)
        fdata.at[i, 'Shape.FractalDimension'] = _fractal_dimension(nprop.image)

        # compute Minor to Major axis ratio
        if nprop.major_axis_length > 0:
            fdata.at[i, 'Shape.MinorMajorAxisRatio'] = \
                nprop.minor_axis_length / nprop.major_axis_length
        else:
            fdata.at[i, 'Shape.MinorMajorAxisRatio'] = 1

        # Hu moments, raw and possibly also intensity-weighted
        fdata.loc[i, hu_cols] = nprop.moments_hu
        if intensity_wtd:
            fdata.loc[i, wtd_hu_cols] = nprop.weighted_moments_hu

    return fdata


def _fractal_dimension(Z):
    """
    Calculate the fractal dimension of an object (boundary complexity).

    Source: https://gist.github.com/rougier/e5eafc276a4e54f516ed5559df4242c0

    From https://en.wikipedia.org/wiki/Minkowski–Bouligand_dimension ...
    In fractal geometry, the Minkowski–Bouligand dimension, also known as
    Minkowski dimension or box-counting dimension, is a way of determining the
    fractal dimension of a set S in a Euclidean space Rn, or more generally in
    a metric space (X, d).

    """
    # Only for 2d binary image
    assert len(Z.shape) == 2
    Z = Z > 0

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(arr, k):
        S = np.add.reduceat(
            np.add.reduceat(arr, np.arange(0, arr.shape[0], k), axis=0),
            np.arange(0, arr.shape[1], k),
            axis=1)
        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k * k))[0])

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2 ** np.floor(np.log(p) / np.log(2))

    # Extract the exponent
    n = int(np.log(n) / np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2 ** np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)

    return -coeffs[0]
