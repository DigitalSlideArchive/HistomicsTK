"""Compute intensity features in labeled image."""
import numpy as np


def compute_intensity_features(
        im_label, im_intensity, num_hist_bins=10,
        rprops=None, feature_list=None):
    """Calculate intensity features from an intensity image.

    Parameters
    ----------
    im_label : array_like
        A labeled mask image wherein intensity of a pixel is the ID of the
        object it belongs to. Non-zero values are considered to be foreground
        objects.

    im_intensity : array_like
        Intensity image.

    num_hist_bins: int, optional
        Number of bins used to computed the intensity histogram of an object.
        Histogram is used to energy and entropy features. Default is 10.

    rprops : output of skimage.measure.regionprops, optional
        rprops = skimage.measure.regionprops( im_label ). If rprops is not
        passed then it will be computed inside which will increase the
        computation time.

    feature_list : list, default is None
        list of intensity features to return.
        If none, all intensity features are returned.

    Returns
    -------
    fdata: pandas.DataFrame
        A pandas dataframe containing the intensity features listed below for
        each object/label.

    Notes
    -----
    List of intensity features computed by this function:

    Intensity.Min : float
        Minimum intensity of object pixels.

    Intensity.Max : float
        Maximum intensity of object pixels.

    Intensity.Mean : float
        Mean intensity of object pixels

    Intensity.Median : float
        Median intensity of object pixels

    Intensity.MeanMedianDiff : float
        Difference between mean and median intensities of object pixels.

    Intensity.Std : float
        Standard deviation of the intensities of object pixels

    Intensity.IQR: float
        Inter-quartile range of the intensities of object pixels

    Intensity.MAD: float
        Median absolute deviation of the intensities of object pixels

    Intensity.Skewness : float
        Skewness of the intensities of object pixels. Value is 0 when all
        intensity values are equal.

    Intensity.Kurtosis : float
        Kurtosis of the intensities of object pixels. Value is -3 when all
        values are equal.

    Intensity.HistEnergy : float
        Energy of the intensity histogram of object pixels

    Intensity.HistEntropy : float
        Entropy of the intensity histogram of object pixels.

    References
    ----------
    .. [#] Daniel Zwillinger and Stephen Kokoska. "CRC standard probability
       and statistics tables and formulae," Crc Press, 1999.

    """
    import pandas as pd
    import scipy.stats
    from skimage.measure import regionprops

    default_feature_list = [
        'Intensity.Min',
        'Intensity.Max',
        'Intensity.Mean',
        'Intensity.Median',
        'Intensity.MeanMedianDiff',
        'Intensity.Std',
        'Intensity.IQR',
        'Intensity.MAD',
        'Intensity.Skewness',
        'Intensity.Kurtosis',
        'Intensity.HistEnergy',
        'Intensity.HistEntropy',
    ]

    # List of feature names
    if feature_list is None:
        feature_list = default_feature_list
    else:
        assert all(j in default_feature_list for j in feature_list), \
            "Some feature names are not recognized."

    # compute object properties if not provided
    if rprops is None:
        rprops = regionprops(im_label)

    # create pandas data frame containing the features for each object
    numFeatures = len(feature_list)
    numLabels = len(rprops)
    fdata = pd.DataFrame(np.zeros((numLabels, numFeatures)),
                         columns=feature_list)

    # conditionally execute calculations if x in the features list
    def _conditional_execution(feature, func, *args, **kwargs):
        if feature in feature_list:
            fdata.at[i, feature] = func(*args, **kwargs)

    def _return_input(x):
        return x

    for i in range(numLabels):

        # get intensities of object pixels
        pixelIntensities = np.sort(
            im_intensity[rprops[i].coords[:, 0], rprops[i].coords[:, 1]]
        )

        # simple descriptors
        meanIntensity = np.mean(pixelIntensities)
        medianIntensity = np.median(pixelIntensities)
        _conditional_execution('Intensity.Min', np.min, pixelIntensities)
        _conditional_execution('Intensity.Max', np.max, pixelIntensities)
        _conditional_execution('Intensity.Mean', _return_input, meanIntensity)
        _conditional_execution(
            'Intensity.Median', _return_input, medianIntensity)
        _conditional_execution(
            'Intensity.MeanMedianDiff', _return_input,
            meanIntensity - medianIntensity)
        _conditional_execution('Intensity.Std', np.std, pixelIntensities)
        _conditional_execution(
            'Intensity.Skewness', scipy.stats.skew, pixelIntensities)
        _conditional_execution(
            'Intensity.Kurtosis', scipy.stats.kurtosis, pixelIntensities)

        # inter-quartile range
        _conditional_execution(
            'Intensity.IQR', scipy.stats.iqr, pixelIntensities)

        # median absolute deviation
        _conditional_execution(
            'Intensity.MAD', np.median,
            np.abs(pixelIntensities - medianIntensity))

        # histogram-based features
        if any(j in feature_list for j in [
                'Intensity.HistEntropy', 'Intensity.HistEnergy']):

            # compute intensity histogram
            hist, bins = np.histogram(pixelIntensities, bins=num_hist_bins)
            prob = hist/np.sum(hist, dtype=np.float32)

            # entropy and energy
            _conditional_execution(
                'Intensity.HistEntropy', scipy.stats.entropy, prob)
            _conditional_execution('Intensity.HistEnergy', np.sum, prob**2)

    return fdata
