import numpy as np
import pandas as pd
import scipy.stats
from skimage.feature import canny
from skimage.measure import regionprops


def ComputeGradientFeatures(im_label, im_intensity,
                            num_hist_bins=10, rprops=None):
    """
    Calculates gradient features from an intensity image.

    Parameters
    ----------
    im_label : array_like
        A labeled mask image wherein intensity of a pixel is the ID of the
        object it belongs to. Non-zero values are considered to be foreground
        objects.

    im_intensity : array_like
        Intensity image

    num_hist_bins: int, optional
        Number of bins used to computed the gradient histogram of an object.
        Histogram is used to energy and entropy features. Default is 10.

    rprops : output of skimage.measure.regionprops, optional
        rprops = skimage.measure.regionprops( im_label ). If rprops is not
        passed then it will be computed inside which will increase the
        computation time.

    Returns
    -------
    fdata: pandas.DataFrame
        A pandas dataframe containing the gradient features listed below for
        each object/label.

    Notes
    -----
    List of gradient features computed by this function:

    Gradient.Mean : float
        Mean of gradient data.
    Gradient.Std : float
        Standard deviation of gradient data.
    Gradient.Skewness : float
        Skewness of gradient data. Value is 0 when all values are equal.
    Gradient.Kurtosis : float
        Kurtosis of gradient data. Value is -3 when all values are equal.
    Gradient.HistEnergy : float
        Energy of the intensity histogram of object pixels
    Gradient.HistEntropy : float
        Entropy of the intensity histogram of object pixels.
    Gradient.SumCanny : float
        Sum of canny filtered gradient data.
    Gradient.MeanCanny : float
        Mean of canny filtered gradient data.

    References
    ----------
    .. [1] Daniel Zwillinger and Stephen Kokoska. "CRC standard probability
    and statistics tables and formulae," Crc Press, 1999.
    """

    # List of feature names
    feature_list = [
        'Gradient.Mean',
        'Gradient.Std',
        'Gradient.Skewness',
        'Gradient.Kurtosis',
        'Gradient.HistEntropy',
        'Gradient.HistEnergy',
        'Gradient.SumCanny',
        'Gradient.MeanCanny',
    ]

    # compute object properties if not provided
    if rprops is None:
        rprops = regionprops(im_label)

    # create pandas data frame containing the features for each object
    numFeatures = len(feature_list)
    numLabels = len(rprops)
    fdata = pd.DataFrame(np.zeros((numLabels, numFeatures)),
                         columns=feature_list)

    Gx, Gy = np.gradient(im_intensity)
    diffG = np.sqrt(Gx**2 + Gy**2)
    cannyG = canny(im_intensity)

    for i in range(numLabels):

        # get gradients of object pixels
        pixelGradients = np.sort(
            diffG[rprops[i].coords[:, 0], rprops[i].coords[:, 1]]
        )

        # compute mean
        fdata.at[i, 'Gradient.Mean'] = np.mean(pixelGradients)

        # compute standard deviation
        fdata.at[i, 'Gradient.Std'] = np.std(pixelGradients)

        # compute skewness
        fdata.at[i, 'Gradient.Skewness'] = scipy.stats.skew(pixelGradients)

        # compute kurtosis
        fdata.at[i, 'Gradient.Kurtosi'] = \
            scipy.stats.kurtosis(pixelGradients)

        # compute intensity histogram
        hist, bins = np.histogram(pixelGradients, bins=num_hist_bins)
        prob = hist/np.sum(hist, dtype=np.float32)

        # compute entropy
        fdata.at[i, 'Gradient.HistEntropy'] = scipy.stats.entropy(prob)

        # compute energy
        fdata.at[i, 'Gradient.HistEnergy'] = np.sum(prob**2)

        bw_canny = cannyG[rprops[i].coords[:, 0], rprops[i].coords[:, 1]]
        fdata.at[i, 'Gradient.SumCanny'] = np.sum(bw_canny)

        fdata.at[i, 'Gradient.MeanCanny'] = \
            fdata.at[i, 'Gradient.SumCanny'] / len(pixelGradients)

    return fdata
