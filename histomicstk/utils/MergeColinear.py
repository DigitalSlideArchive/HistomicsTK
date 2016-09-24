import numpy as np


def MergeColinear(X, Y):
    """Processes boundary coordinates in polyline with vertices X, Y to remove
    redundant colinear points. Polyline is not assumed to be open or closed.

    Parameters
    ----------
    X : array_like
        One dimensional array of horizontal boundary coordinates.
    Y : array_like
        One dimensional array of vertical boundary coordinates.

    Returns
    -------
    XOut : array_like
        X with colinear boundary points removed.
    YOUt : array_like
        Y with colinear boundary points removed.
    """

    # compute boundary differences
    dX = np.diff(X)
    dY = np.diff(Y)

    # detect and delete stationary repeats
    Repeats = np.argwhere((dX == 0) & (dY == 0))
    np.delete(X, Repeats)
    np.delete(Y, Repeats)
    np.delete(dX, Repeats)
    np.delete(dY, Repeats)

    # calculate slope transitions
    slope = dY / dX

    # find transitions
    dslope = np.diff(slope)
    dslope[np.isnan(dslope)] = 0
    transitions = np.argwhere(dslope != 0)

    # construct merged sequences
    XOut = np.append(X[0], X[transitions+1])
    YOut = np.append(Y[0], Y[transitions+1])
    XOut = np.append(XOut, X[-1])
    YOut = np.append(YOut, Y[-1])

    return XOut, YOut
