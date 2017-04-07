import numpy as np
from skimage.measure import regionprops

from ._trace_object_boundaries_cython import _trace_object_boundaries_cython


def trace_object_boundaries(im_label,
                            conn=4, trace_all=False,
                            x_start=None, y_start=None, max_length=None):
    """Performs exterior boundary tracing of one or more objects in a label
    mask. If a starting point is not provided then a raster scan will be performed
    to identify the starting pixel.

    Parameters
    ----------
    im_label : array_like
        A binary mask image.
    conn : int
        Neighborhood connectivity to evaluate. Valid values are 4 or 8.
        Default value = 4.
    trace_all : bool
        Specify True if you want to trace boundaries of all objects.
        Default = False
    x_start : int
        Starting horizontal coordinate to begin tracing. Default value = None.
    y_start : int
        Starting vertical coordinate to begin tracing. Default value = None.
    max_length : int
        Maximum boundary length to trace before terminating. Default value =
        None.

    Notes
    -----
    The Improved Simple Boundary Follower (ISBF) from the reference below is
    used for 4-connected tracing. This algorithm provides accurate tracing with
    competitive execution times. 8-connected tracing is implemented using the
    Moore tracing algorithm.

    Returns
    -------
    X : array_like
        A set of 1D array of horizontal coordinates of contour seed pixels for
        tracing.
    Y : array_like
        A set of 1D array of the vertical coordinates of seed pixels for
        tracing.

    References
    ----------
    .. J. Seo et al "Fast Contour-Tracing Algorithm Based on a Pixel-
       Following Method for Image Sensors" in Sensors,vol.16,no.353,
       doi:10.3390/s16030353, 2016.

    """

    if max_length is None:
        max_length = float('inf')

    X = []
    Y = []

    if trace_all:

        rprops = regionprops(im_label)
        numLabels = len(rprops)

        x_start = -1
        y_start = -1

        for i in range(numLabels):

            # get bounds of label mask
            min_row, min_col, max_row, max_col = rprops[i].bbox

            # grab label mask
            lmask = (
                im_label[
                    min_row:max_row, min_col:max_col
                ] == rprops[i].label
            ).astype(np.bool)

            mrows = max_row - min_row + 2
            mcols = max_col - min_col + 2

            mask = np.zeros((mrows, mcols))
            mask[1:mrows-1, 1:mcols-1] = lmask

            by, bx = _trace_object_boundaries_cython(
                np.ascontiguousarray(
                    mask, dtype=np.int), conn, x_start, y_start, max_length
            )

            bx = bx + min_row - 1
            by = by + min_col - 1

            X.append(bx)
            Y.append(by)

    else:

        rprops = regionprops(im_label.astype(int))
        numLabels = len(rprops)

        if numLabels > 1:
            raise ValueError("Number of labels should be 1 !!")

        if (x_start is None and y_start is not None) | \
                (x_start is not None and y_start is None):
            raise ValueError("x_start or y_start is not defined !!")

        if x_start is None and y_start is None:
            x_start = -1
            y_start = -1

        by, bx = _trace_object_boundaries_cython(
            np.ascontiguousarray(
                im_label, dtype=np.int), conn, x_start, y_start, max_length
        )

        X.append(bx)
        Y.append(by)

    return X, Y
