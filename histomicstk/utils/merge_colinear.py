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

    # detect and delete points that are the same as the following point.
    Repeats = np.argwhere((np.diff(x) == 0) & (np.diff(y) == 0))
    xout = np.delete(x, Repeats)
    yout = np.delete(y, Repeats)

    # Calculating the slope for each transition could involve division by
    # zero, so we instead detect colinearity by noting that two non-zero
    # vectors are colinear if and only if the cross product of the vectors
    # is zero.  We convert to signed floats for two reasons: (1) in case
    # the inputs are unsigned values, (2) in case the cross product is
    # large.
    dX = np.diff(xout.astype(np.float64))
    dY = np.diff(yout.astype(np.float64))
    colinear = np.argwhere(dX[:-1] * dY[1:] == dX[1:] * dY[:-1]) + 1
    xout = np.delete(xout, colinear)
    yout = np.delete(yout, colinear)

    return xout, yout
