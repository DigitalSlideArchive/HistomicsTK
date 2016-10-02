import numpy as np
import ctypes
from .trace_object_boundary import trace_object_boundary


def trace_boundary(Mask, Connectivity=4, XStart=None, YStart=None,
                   MaxLength=np.inf):
    """Performs exterior boundary tracing of a single object in a binary mask.
    If a starting point is not provided then a raster scan will be performed to
    identify the starting pixel.

    Parameters:
    -----------
    Mask : array_like
        A boolean type image where foreground pixels have value 'True', and
        background pixels have value 'False'.
    Connectivity : int
        Neighborhood connectivity to evaluate. Valid values are 4 or 8.
        Default value = 4.
    XStart : int
        Starting horizontal coordinate to begin tracing. Default value = None.
    YStart : int
        Starting vertical coordinate to begin tracing. Default value = None.
    MaxLength : int
        Maximum boundary length to trace before terminating. Default value =
        np.inf.

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

    # check type of input mask
    if Mask.dtype != np.dtype('bool'):
        raise TypeError("Input 'Mask' must be a bool")

    # scan for starting pixel if none provided
    if XStart is None and YStart is None:
        Indices = np.nonzero(Mask)
        if Indices[0].size > 0:
            YStart = Indices[0][0]
            XStart = Indices[1][0]
        else:
            X = np.array([], dtype=np.uint32)
            Y = np.array([], dtype=np.uint32)
            return X, Y

    Mask = np.ascontiguousarray(Mask, dtype=ctypes.c_int)

    X, Y = trace_object_boundary(Mask, Connectivity, XStart, YStart,
                                 MaxLength)

    # convert outputs from list to numpy array
    X = np.array(X, dtype=np.uint32)
    Y = np.array(Y, dtype=np.uint32)

    return X, Y
