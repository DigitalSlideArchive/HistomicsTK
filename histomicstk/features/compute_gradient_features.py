import numpy as np


def compute_gradient_features(im_label, im_intensity,
                              num_hist_bins=10, rprops=None):
    """Calculates gradient features from an intensity image.

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

    Gradient.Mag.Mean : float
        Mean of gradient data.

    Gradient.Mag.Std : float
        Standard deviation of gradient data.

    Gradient.Mag.Skewness : float
        Skewness of gradient data. Value is 0 when all values are equal.

    Gradient.Mag.Kurtosis : float
        Kurtosis of gradient data. Value is -3 when all values are equal.

    Gradient.Mag.HistEnergy : float
        Energy of the gradient magnitude histogram of object pixels

    Gradient.Mag.HistEnergy : float
        Entropy of the gradient magnitude histogram of object pixels.

    Gradient.Canny.Sum : float
        Sum of canny filtered gradient data.

    Gradient.Canny.Mean : float
        Mean of canny filtered gradient data.

    References
    ----------
    .. [#] Daniel Zwillinger and Stephen Kokoska. "CRC standard probability
       and statistics tables and formulae," Crc Press, 1999.

    """
    import pandas as pd
    import scipy.stats
    from skimage.feature import canny
    from skimage.measure import regionprops

    # List of feature names
    feature_list = [
        'Gradient.Mag.Mean',
        'Gradient.Mag.Std',
        'Gradient.Mag.Skewness',
        'Gradient.Mag.Kurtosis',
        'Gradient.Mag.HistEntropy',
        'Gradient.Mag.HistEnergy',
        'Gradient.Canny.Sum',
        'Gradient.Canny.Mean',
    ]

    # Compute object properties if not provided
    if rprops is None:
        rprops = regionprops(im_label)

    numLabels = len(rprops)

    Gx, Gy = np.gradient(im_intensity)
    diffG = np.sqrt(Gx**2 + Gy**2)
    cannyG = canny(im_intensity)

    # Prepare data collection
    data = []

    for i in range(numLabels):
        if rprops[i] is None:
            continue

        # get gradients of object pixels
        pixelGradients = np.sort(diffG[rprops[i].coords[:, 0], rprops[i].coords[:, 1]])

        # Compute intensity histogram
        hist, bins = np.histogram(pixelGradients, bins=num_hist_bins)
        prob = hist / np.sum(hist, dtype=np.float32)

        # Canny edges for the object
        bw_canny = cannyG[rprops[i].coords[:, 0], rprops[i].coords[:, 1]]
        canny_sum = np.sum(bw_canny).astype('float')

        # Aggregate features
        features = [
            np.mean(pixelGradients),  # Mean
            np.std(pixelGradients),  # Std
            scipy.stats.skew(pixelGradients),  # Skewness
            scipy.stats.kurtosis(pixelGradients),  # Kurtosis
            scipy.stats.entropy(prob),  # HistEntropy
            np.sum(prob**2),  # HistEnergy
            canny_sum,  # Canny.Sum
            canny_sum / len(pixelGradients),  # Canny.Mean
        ]

        data.append(features)

    # Create DataFrame
    feature_list = [
        'Gradient.Mag.Mean',
        'Gradient.Mag.Std',
        'Gradient.Mag.Skewness',
        'Gradient.Mag.Kurtosis',
        'Gradient.Mag.HistEntropy',
        'Gradient.Mag.HistEnergy',
        'Gradient.Canny.Sum',
        'Gradient.Canny.Mean',
    ]

    fdata = pd.DataFrame(data, columns=feature_list)

    return fdata
