import collections
import numpy as np
from skimage.segmentation import find_boundaries


def ComputeFSDs(Label, K=128, Fs=6):
    """
    Calculates `Fourier shape descriptors` from a label mask.

    Parameters
    ----------
    Label : array_like
        A M x N label mask.
    K : Number of points for boundary resampling to calculate fourier
        descriptors. Default value = 128.
    Fs : Number of frequency bins for calculating FSDs. Default value = 6.

    Returns
    -------
    FSDGroup : array_like
        A 1 x Fs FSD arrays.

    References
    ----------
    .. [1] D. Zhang et al. "A comparative study on shape retrieval using
       Fourier descriptors with different shape signatures," In Proc.
       ICIMADE01, 2001.
    """

    # initialize FSD feature group
    FSDGroup = np.zeros(Fs)

    # find nucleus boundaries
    Bounds = np.argwhere(
        find_boundaries(Label, mode="inner").astype(np.uint8) == 1
    )

    # fourier descriptors, spaced evenly over the interval 1:K/2
    Interval = np.round(
        np.power(
            2, np.linspace(0, np.log2(K)-1, Fs+1, endpoint=True)
        )
    ).astype(np.uint8)

    FSDGroup = FSDs(Bounds[:, 0], Bounds[:, 1], K, Interval)

    return FSDGroup


def InterpolateArcLength(X, Y, L):
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


def FSDs(X, Y, K, Intervals):
    """
    Calculated FSDs from boundary points X,Y. Boundaries are resampled to
    have K equally spaced points (arclength) around the shape. The curvature
    is calculated using the cumulative angular function, measuring the
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
    iXY = InterpolateArcLength(X, Y, K)
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
