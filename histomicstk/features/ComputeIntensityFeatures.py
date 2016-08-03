import numpy as np
import pandas as pd
import scipy.stats
from skimage.measure import regionprops


def ComputeIntensityFeatures(im_label, im_intensity,
                             num_hist_bins=10, rprops=None):
    """
    Calculates intensity features from an intensity image.

    Parameters
    ----------
    im_label : array_like
        A labeled mask image wherein intensity of a pixel is the ID of the
        object it belongs to. Non-zero values are considered to be foreground
        objects.

    im_intensity : array_like
        Intensity image

    num_hist_bins: int, optional
        Number of bins used to computed the intensity histogram of an object.
        Histogram is used to energy and entropy features. Default is 10.

    rprops : output of skimage.measure.regionprops, optional
        rprops = skimage.measure.regionprops( im_label ). If rprops is not
        passed then it will be computed inside which will increase the
        computation time.

    Coords : array_like
        A N x 2 coordinate list of a region.

    Returns
    -------
    fdata: pandas.DataFrame
        A pandas dataframe containing the intensity features listed below for
        each object/label.

    Notes
    -----
    List of intensity features computed by this function:

    MinIntensity : float
        Minimum intensity of object pixels.

    MaxIntensity : float
        Maximum intensity of object pixels.

    MeanIntensity : float
        Mean intensity of object pixels

    MedianIntensity : float
        Median intensity of object pixels

    MeanMedianDifferenceIntensity : float
        Difference between mean and median intensities of object pixels.

    StdIntensity : float
        Standard deviation of the intensities of object pixels

    IqrIntensity: float
        Inter-quartile range of the intensities of object pixels

    MadIntensity: float
        Median absolute deviation of the intensities of object pixels

    SkewnessIntensity : float
        Skewness of the intensities of object pixels. Value is 0 when all
        intensity values are equal.

    KurtosisIntensity : float
        Kurtosis of the intensities of object pixels. Value is -3 when all
        values are equal.

    HistogramEnergy : float
        Energy of the intensity histogram of object pixels

    HistogramEntropy : float
        Entropy of the intensity histogram of object pixels.
    """

    # List of feature names in alphabetical order
    feature_list = [
        'MinIntensity',
        'MaxIntensity',
        'MeanIntensity',
        'MedianIntensity',
        'MeanMedianDifferenceIntensity',
        'StdIntensity',
        'IqrIntensity',
        'MadIntensity',
        'SkewnessIntensity',
        'KurtosisIntensity',
        'HistogramEnergy',
        'HistogramEntropy',
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

        # get intensities of object pixels
        pixelIntensities = np.sort(
            im_intensity[rprops[i].Coords[:, 0], rprops[i].Coords[:, 1]]
        )

        # compute min
        fdata.at[i, 'MinIntensity'] = np.min(pixelIntensities)

        # compute max
        fdata.at[i, 'MaxIntensity'] = np.max(pixelIntensities)

        # compute mean
        meanIntensity = np.mean(pixelIntensities)
        fdata.at[i, 'MeanIntensity'] = meanIntensity

        # compute median
        medianIntensity = np.median(pixelIntensities)
        fdata.at[i, 'MedianIntensity'] = medianIntensity

        # compute mean median differnece
        fdata.at[i, 'MeanMedianDifferenceIntensity'] = \
            meanIntensity - medianIntensity

        # compute standard deviation
        fdata.at[i, 'StdIntensity'] = np.std(pixelIntensities)

        # compute inter-quartile range
        fdata.at[i, 'IqrIntensity'] = scipy.stats.iqr(pixelIntensities)

        # compute skewness
        fdata.at[i, 'SkewnessIntensity'] = scipy.stats.skew(pixelIntensities)

        # compute kurtosis
        fdata.at[i, 'KurtosisIntensity'] = \
            scipy.stats.kurtosis(pixelIntensities)

        # compute intensity histogram
        hist, bins = np.histogram(pixelIntensities, bins=num_hist_bins)
        prob = hist/np.sum(hist, dtype=np.float32)

        # compute entropy
        fdata.at[i, 'HistogramEntropy'] = scipy.stats.entropy(prob)

        # compute energy
        fdata.at[i, 'HistogramEnergy'] = np.sum(prob**2)

    return fdata
