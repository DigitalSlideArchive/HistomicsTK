import numpy as np


def TraceBounds(Mask, Connectivity=4, XStart=None, YStart=None,
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

    See Also:
    ---------

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
            X = []
            Y = []
            return X, Y

    # choose algorithm based on connectivity
    if Connectivity == 4:
        X, Y = ISBF(Mask, XStart, YStart, MaxLength)
    elif Connectivity == 8:
        X, Y = Moore(Mask, XStart, YStart, MaxLength)
    else:
        raise ValueError("Input 'Connectivity' must be 4 or 8.")

    return X, Y


def Moore(Mask, XStart, YStart, MaxLength):
    """Performs exterior boundary tracing of a single object in a binary mask
    using the Moore-neighbor algorithm.

    Parameters:
    -----------
    Mask : array_like
        A boolean type image where foreground pixels have value 'True', and
        background pixels have value 'False'.
    XStart : int
        Starting horizontal coordinate to begin tracing.
    YStart : int
        Starting vertical coordinate to begin tracing.
    MaxLength : int
        Maximum boundary length to trace before terminating.

    Returns:
    --------
    X : array_like
        A 1D array of horizontal coordinates of contour seed pixels for
        tracing.
    Y : array_like
        A 1D array of the vertical coordinates of seed pixels for tracing.
    """

    # initialize outputs
    X = []
    Y = []

    # add starting pixel and direction to outputs
    X.append(XStart)
    Y.append(YStart)

    # initialize direction
    DX = 1
    DY = 0

    # define clockwise ordered indices
    row = [2, 1, 0, 0, 0, 1, 2, 2]
    col = [0, 0, 0, 1, 2, 2, 2, 1]
    dX = [-1, 0, 0, 1, 1, 0, 0, -1]
    dY = [0, -1, -1, 0, 0, 1, 1, 0]
    oX = [-1, -1, -1, 0, 1, 1, 1, 0]
    oY = [1, 0, -1, -1, -1, 0, 1, 1]

    while True:

        # rotate template surrounding current location to fit relative frame
        if (DX == 1) & (DY == 0):
            T = np.rot90(Mask[Y[-1]-1:Y[-1]+2, X[-1]-1:X[-1]+2], 1)
            Angle = np.pi/2
        elif (DX == 0) & (DY == -1):
            T = Mask[Y[-1]-1:Y[-1]+2, X[-1]-1:X[-1]+2]
            Angle = 0
        elif (DX == -1) & (DY == 0):
            T = np.rot90(Mask[Y[-1]-1:Y[-1]+2, X[-1]-1:X[-1]+2], 3)
            Angle = 3 * np.pi / 2
        else:  # (Direction[0] == 0) & (DY[-1] == 1):
            T = np.rot90(Mask[Y[-1]-1:Y[-1]+2, X[-1]-1:X[-1]+2], 2)
            Angle = np.pi

        # get first template entry that is 1
        Move = np.argmax(T[row, col])

        # transform points by incoming directions and add to contours
        R = np.array([[np.cos(Angle), -np.sin(Angle)],
                      [np.sin(Angle), np.cos(Angle)]])
        Coords = R.dot(np.vstack((np.array(oX[Move]),
                                  np.array(oY[Move])))).round()
        Direction = R.dot(np.vstack((dX[Move], dY[Move]))).round()
        DX = Direction[0]
        DY = Direction[1]

        # capture next location
        X.append(X[-1] + Coords[0][0])
        Y.append(Y[-1] + Coords[1][0])

        # check of last two points on contour are first two points on contour
        if(len(X) > 3):
            if(len(X) >= MaxLength) or \
                (X[-1] == X[1] and X[-2] == X[0] and
                 Y[-1] == Y[1] and Y[-2] == Y[0]):
                    X = X[0:-1]
                    Y = Y[0:-1]
                    break

    return X, Y


def ISBF(Mask, XStart, YStart, MaxLength):  # noqa: C901
    """Performs exterior boundary tracing of a single object in a binary mask
    using the Improved Simple Boundary Follower (ISBF) algorithm from the
    reference below.

    Parameters:
    -----------
    Mask : array_like
        A boolean type image where foreground pixels have value 'True', and
        background pixels have value 'False'.
    XStart : int
        Starting horizontal coordinate to begin tracing.
    YStart : int
        Starting vertical coordinate to begin tracing.
    MaxLength : int
        Maximum boundary length to trace before terminating.

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

    # initialize outputs
    X = []
    Y = []

    # add starting pixel and direction to outputs
    X.append(XStart)
    Y.append(YStart)

    # initialize direction
    DX = 1
    DY = 0

    while True:

        # rotate template surrounding current location to fit relative frame
        if (DX == 1) & (DY == 0):
            T = np.rot90(Mask[Y[-1]-1:Y[-1]+1, X[-1]-1:X[-1]+2], 1)
            Angle = np.pi/2
        elif (DX == 0) & (DY == -1):
            T = Mask[Y[-1]-1:Y[-1]+2, X[-1]-1:X[-1]+1]
            Angle = 0
        elif (DX == -1) & (DY == 0):
            T = np.rot90(Mask[Y[-1]:Y[-1]+2, X[-1]-1:X[-1]+2], 3)
            Angle = 3 * np.pi / 2
        else:  # (Direction[0] == 0) & (DY[-1] == 1):
            T = np.rot90(Mask[Y[-1]-1:Y[-1]+2, X[-1]:X[-1]+2], 2)
            Angle = np.pi

        # initialize contours
        cX = []
        cY = []

        if T[1, 0]:
            # 'left' neighbor
            cX.append(-1)
            cY.append(0)
            DX = -1
            DY = 0
        else:
            if T[2, 0] and not T[2, 1]:
                # inner-outer corner at left-rear
                cX.append(-1)
                cY.append(+1)
                DX = 0
                DY = 1
            else:
                if T[0, 0]:
                    if T[0, 1]:
                        # inner corner at front
                        cX.append(0)
                        cY.append(-1)
                        cX.append(-1)
                        cY.append(0)
                        DX = 0
                        DY = -1
                    else:
                        # inner-outer corner at front-left
                        cX.append(-1)
                        cY.append(-1)
                        DX = 0
                        DY = -1
                elif T[0, 1]:
                    # front neighbor
                    cX.append(0)
                    cY.append(-1)
                    DX = 1
                    DY = 0
                else:
                    # outer corner
                    DX = 0
                    DY = 1

        # transform points by incoming directions and add to contours
        R = np.array([[np.cos(Angle), -np.sin(Angle)],
                      [np.sin(Angle), np.cos(Angle)]])
        Coords = R.dot(np.vstack((np.array(cX), np.array(cY)))).round()
        Direction = R.dot(np.vstack((DX, DY))).round()
        DX = Direction[0]
        DY = Direction[1]

        for i in np.arange(Coords.shape[1]):
            X.append(X[-1] + Coords[0, i])
            Y.append(Y[-1] + Coords[1, i])

        # check of last two points on contour are first two points on contour
        if(len(X) > 3):
            if(len(X) >= MaxLength) or \
                (X[-1] == X[1] and X[-2] == X[0] and
                 Y[-1] == Y[1] and Y[-2] == Y[0]):
                    X = X[0:-1]
                    Y = Y[0:-1]
                    break

            # check addtional points if needed
            if Coords.shape[1] == 2:
                if(X[-2] == X[1] and X[-3] == X[0] and
                   Y[-2] == Y[1] and Y[-3] == Y[0]):
                        X = X[0:-2]
                        Y = Y[0:-2]
                        break

    return X, Y
