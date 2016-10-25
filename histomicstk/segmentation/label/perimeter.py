import numpy as np


def perimeter(L, Connectivity=4):
    """Converts a label or binary mask image to a binary perimeter image.

    Uses 4-neighbor or 8-neighbor shifts to detect pixels whose values do
    not agree with their neighbors.

    Parameters
    ----------
    L : array_like
        A label or binary mask image.
    Connectivity : double or int
        Neighborhood connectivity to evaluate. Valid values are 4 or 8.
        Default value = 4.

    Returns
    -------
    Mask : array_like
        A binary image where object perimeter pixels have value 1, and
        non-perimeter pixels have value 0.

    See Also
    --------
    histomicstk.segmentation.embed_boundaries
    """

    # initialize temporary variable
    Mask = np.zeros(L.shape)
    Temp = np.zeros(L.shape)

    # check left-right neighbors
    Temp[:, 0:-2] = np.not_equal(L[:, 0:-2], L[:, 1:-1])
    Temp[:, 1:-1] = np.logical_or(Temp[:, 1:-1], Temp[:, 0:-2])
    Mask = np.logical_or(Mask, Temp)

    # check up-down neighbors
    Temp[0:-2, :] = np.not_equal(L[0:-2, :], L[1:-1, :])
    Temp[1:-1, :] = np.logical_or(Temp[1:-1, :], Temp[0:-2, :])
    Mask = np.logical_or(Mask, Temp)

    # additional calculations if Connectivity == 8
    if(Connectivity == 8):

        # slope 1 diagonal shift
        Temp[1:-1, 0:-2] = np.not_equal(L[0:-2, 1:-2], L[1:-1, 0:-2])
        Temp[0:-2, 1:-1] = np.logical_or(Temp[0:-2, 1:-1], Temp[1:-1, 0:-2])
        Mask = np.logical_or(Mask, Temp)

        # slope -1 diagonal shift
        Temp[1:-1, 1:-1] = np.not_equal(L[0:-2, 0:-2], L[1:-1, 1:-1])
        Temp[0:-2, 0:-2] = np.logical_or(Temp[0:-2, 0:-2], Temp[1:-1, 1:-1])
        Mask = np.logical_or(Mask, Temp)

    # generate label-valued output
    return Mask.astype(np.uint32) * L
