import numpy as np


def merge_colinear(x, y):
    """Processes boundary coordinates in polyline with vertices X, Y to remove
    redundant colinear points. Polyline is not assumed to be open or closed.

    Parameters
    ----------
    x : array_like
        One dimensional array of horizontal boundary coordinates.
    y : array_like
        One dimensional array of vertical boundary coordinates.

    Returns
    -------
    xout : array_like
        X with colinear boundary points removed.
    yout : array_like
        Y with colinear boundary points removed.

    """

    # compute boundary differences
    dX = np.diff(x)
    dY = np.diff(y)

    # detect and delete stationary repeats
    Repeats = np.argwhere((dX == 0) & (dY == 0))
    np.delete(x, Repeats)
    np.delete(y, Repeats)
    np.delete(dX, Repeats)
    np.delete(dY, Repeats)

    # calculate slope transitions
    slope = dY / dX

    # find transitions
    dslope = np.diff(slope)
    dslope[np.isnan(dslope)] = 0
    transitions = np.argwhere(dslope != 0)

    # construct merged sequences
    xout = np.append(x[0], x[transitions + 1])
    yout = np.append(y[0], y[transitions + 1])
    xout = np.append(xout, x[-1])
    yout = np.append(yout, y[-1])

    return xout, yout
