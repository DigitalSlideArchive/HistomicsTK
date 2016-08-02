import collections
import numpy as np
import pandas as pd
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries


def ComputeFSDs(im_label, K=128, Fs=6, Delta=8):
    """
    Calculates `Fourier shape descriptors` for each objects.

    Parameters
    ----------
    im_label : array_like
        A labeled mask image wherein intensity of a pixel is the ID of the
        object it belongs to. Non-zero values are considered to be foreground
        objects.
    K : int
        Number of points for boundary resampling to calculate fourier
        descriptors. Default value = 128.
    Fs : int
        Number of frequency bins for calculating FSDs. Default value = 6.
    Delta : int
        Used to dilate nuclei and define cytoplasm region. Default value = 8.

    Returns
    -------
    fdata: Pandas data frame containing the FSD features for each
           object/label.

    References
    ----------
    .. [1] D. Zhang et al. "A comparative study on shape retrieval using
       Fourier descriptors with different shape signatures," In Proc.
       ICIMADE01, 2001.
    """
    # List of feature names
    feature_list = []
    for i in range(0, Fs):
        feature_list = np.append(feature_list, 'FSD' + str(i+1))

    # get Label size x
    sizex = im_label.shape[0]
    sizey = im_label.shape[1]

    # get the number of objects in Label
    rprops = regionprops(im_label)

    numFeatures = len(feature_list)
    numLabels = len(rprops)
    fdata = pd.DataFrame(np.zeros((numLabels, numFeatures)),
                         columns=feature_list)

    # fourier descriptors, spaced evenly over the interval 1:K/2
    Interval = np.round(
        np.power(
            2, np.linspace(0, np.log2(K)-1, Fs+1, endpoint=True)
        )
    ).astype(np.uint8)

    for i in range(numLabels):
        # get bounds of dilated nucleus
        min_row, max_row, min_col, max_col = \
            _GetBounds(rprops[i].bbox, Delta, sizex, sizey)
        # grab label mask
        lmask = (
            im_label[min_row:max_row, min_col:max_col] == rprops[i].label
        ).astype(np.bool)
        # find boundaries
        Bounds = np.argwhere(
            find_boundaries(lmask, mode="inner").astype(np.uint8) == 1
        )
        # compute fourier descriptors
        fdata.at[i, :] = _FSDs(Bounds[:, 0], Bounds[:, 1], K, Interval)

    return fdata


def _InterpolateArcLength(X, Y, L):
    """
    Resamples boundary points [X, Y] at L total equal arc-length locations.

    Parameters
    ----------
    X : array_like
        x points of boundaries
    Y : array_like
        y points of boundaries
    L : int
        Number of points for boundary resampling to calculate fourier
        descriptors. Default value = 128.

    Returns
    -------
    iX : array_like
        L-length vector of horizontal interpolated coordinates with equal
        arc-length spacing.
    iY : array_like
        L-length vector of vertical interpolated coordinates with equal
        arc-length spacing.
    Notes
    -----
    Return values are returned as a namedtuple.
    """

    # length of X
    K = len(X)
    # initialize iX, iY
    iX = np.zeros((0,))
    iY = np.zeros((0,))
    # generate spaced points
    Interval = np.linspace(0, 1, L)
    # get segment lengths
    Lengths = np.sqrt(
        np.power(np.diff(X), 2) + np.power(np.diff(Y), 2)
    )
    # check Lengths
    if Lengths.size:
        # normalize to unit length
        Lengths = Lengths / Lengths.sum()
        # calculate cumulative length along boundary
        Cumulative = np.hstack((0., np.cumsum(Lengths)))
        # place points in 'Interval' along boundary
        Locations = np.digitize(Interval, Cumulative)
        # clip to ends
        Locations[Locations < 1] = 1
        Locations[Locations >= K] = K - 1
        Locations = Locations - 1
        # linear interpolation
        Lie = np.divide(
            (Interval - [Cumulative[i] for i in Locations]),
            [Lengths[i] for i in Locations]
        )
        tX = np.array([X[i] for i in Locations])
        tY = np.array([Y[i] for i in Locations])
        iX = tX + np.multiply(
            np.array([X[i+1] for i in Locations]) - tX, Lie
        )
        iY = tY + np.multiply(
            np.array([Y[i+1] for i in Locations]) - tY, Lie
        )
    iXY = collections.namedtuple('iXY', ['iX', 'iY'])
    Output = iXY(iX, iY)

    return Output


def _FSDs(X, Y, K, Intervals):
    """
    Calculated FSDs from boundary points X,Y. Boundaries are resampled to have
    K equally spaced points (arclength) around the shape. The curvature is
    calculated using the cumulative angular function, measuring the
    displacement of the tangent angle from the starting point of the boundary.
    The K-length fft of the cumulative angular function is calculated, and
    then the elements of 'F' are summed as the spectral energy over
    'Intervals'.

    Parameters
    ----------
    X : array_like
        x points of boundaries
    Y : array_like
        y points of boundaries
    K : int
        Number of points for boundary resampling to calculate fourier
        descriptors. Default value = 128.
    Intervals : array_like
        Intervals spaced evenly over 1:K/2.

    Returns
    -------
    F : array_like
        length(Intervals) vector containing spectral energy of
        cumulative angular function, summed over defined 'Intervals'.
    """

    # check input 'Intervals'
    if Intervals[0] != 1.:
        Intervals = np.hstack((1., Intervals))
    if Intervals[-1] != (K / 2):
        Intervals = np.hstack((Intervals, float(K)))
    # get length of intervals
    L = len(Intervals)
    # initialize F
    F = np.zeros((L-1, )).astype(float)
    # interpolate boundaries
    iXY = _InterpolateArcLength(X, Y, K)
    # check if iXY.iX is not empty
    if iXY.iX.size:
        # calculate curvature
        Curvature = np.arctan2(
            (iXY.iY[1:] - iXY.iY[:-1]),
            (iXY.iX[1:] - iXY.iX[:-1])
        )
        # make curvature cumulative
        Curvature = Curvature - Curvature[0]
        # calculate FFT
        fX = np.fft.fft(Curvature).T
        # spectral energy
        fX = fX * fX.conj()
        fX = fX / fX.sum()
        # calculate 'F' values
        for i in range(L-1):
            F[i] = np.round(
                fX[Intervals[i]-1:Intervals[i+1]].sum(), L
            ).real.astype(float)

    return F


def _GetBounds(bbox, delta, M, N):
    """
    Returns bounds of object in global label image.

    Parameters
    ----------
    bbox : tuple
        Bounding box (min_row, min_col, max_row, max_col).
    delta : int
        Used to dilate nuclei and define cytoplasm region.
        Default value = 8.
    M : int
        X size of label image.
    N : int
        Y size of label image.

    Returns
    -------
    min_row : int
        Minum row of the region bounds.
    max_row : int
        Maximum row of the region bounds.
    min_col : int
        Minum column of the region bounds.
    max_col : int
        Maximum column of the region bounds.
    """

    min_row, min_col, max_row, max_col = bbox

    min_row_out = max(0, (min_row - delta))
    max_row_out = min(M-1, (max_row + delta))
    min_col_out = max(0, (min_col - delta))
    max_col_out = min(N-1, (max_col + delta))

    return min_row_out, max_row_out, min_col_out, max_col_out
