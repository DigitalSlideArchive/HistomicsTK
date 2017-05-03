import numpy as np


def perimeter(im_label, conn=4):
    """Converts a label or binary mask image to a binary perimeter image.

    Uses 4-neighbor or 8-neighbor shifts to detect pixels whose values do
    not agree with their neighbors.

    Parameters
    ----------
    im_label : array_like
        A label or binary mask image.
    conn : double or int
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
    Mask = np.zeros(im_label.shape)
    Temp = np.zeros(im_label.shape)

    # check left-right neighbors
    Temp[:, 0:-2] = np.not_equal(im_label[:, 0:-2], im_label[:, 1:-1])
    Temp[:, 1:-1] = np.logical_or(Temp[:, 1:-1], Temp[:, 0:-2])
    Mask = np.logical_or(Mask, Temp)

    # check up-down neighbors
    Temp[0:-2, :] = np.not_equal(im_label[0:-2, :], im_label[1:-1, :])
    Temp[1:-1, :] = np.logical_or(Temp[1:-1, :], Temp[0:-2, :])
    Mask = np.logical_or(Mask, Temp)

    # additional calculations if conn == 8
    if(conn == 8):

        # slope 1 diagonal shift
        Temp[1:-1, 0:-2] = np.not_equal(im_label[0:-2, 1:-2], im_label[1:-1, 0:-2])
        Temp[0:-2, 1:-1] = np.logical_or(Temp[0:-2, 1:-1], Temp[1:-1, 0:-2])
        Mask = np.logical_or(Mask, Temp)

        # slope -1 diagonal shift
        Temp[1:-1, 1:-1] = np.not_equal(im_label[0:-2, 0:-2], im_label[1:-1, 1:-1])
        Temp[0:-2, 0:-2] = np.logical_or(Temp[0:-2, 0:-2], Temp[1:-1, 1:-1])
        Mask = np.logical_or(Mask, Temp)

    # generate label-valued output
    return Mask.astype(np.uint32) * im_label
