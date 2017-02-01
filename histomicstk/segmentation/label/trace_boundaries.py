from .trace_boundaries_cython import trace_boundaries_cython

import numpy as np

def trace_boundaries(im_label,
                     conn=4, trace_all=False,
                     x_start=None, y_start=None, max_length=None):
    """Performs exterior boundary tracing of one or more objects in a label
    mask. If a starting point is not provided then a raster scan will be performed
    to identify the starting pixel.

    Parameters:
    -----------
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

    Notes:
    ------
    The Improved Simple Boundary Follower (ISBF) from the reference below is
    used for 4-connected tracing. This algorithm provides accurate tracing with
    competitive execution times. 8-connected tracing is implemented using the
    Moore tracing algorithm.

    Returns:
    --------
    X : array_like
        A 1D array of horizontal coordinates of contour seed pixels for
        tracing.
    Y : array_like
        A 1D array of the vertical coordinates of seed pixels for tracing.

    References:
    -----------
    .. [1] J. Seo et al "Fast Contour-Tracing Algorithm Based on a Pixel-
    Following Method for Image Sensors" in Sensors,vol.16,no.353,
    doi:10.3390/s16030353, 2016.
    """

    return trace_boundaries_cython(
        np.ascontiguousarray(im_label, dtype=np.int),
        conn, trace_all, x_start, y_start, max_length)
