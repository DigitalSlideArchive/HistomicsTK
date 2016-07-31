import numpy as np
from histomicstk.segmentation import label as lb
import scipy.ndimage.measurements as ms
from .TraceBounds import TraceBounds


def TraceLabel(Label, Connectivity=4):
    """Performs exterior boundary tracing of a multiple objects in a label
    image.

    Parameters:
    -----------
    Mask : array_like
        A boolean type image where foreground pixels have value 'True', and
        background pixels have value 'False'.
    Connectivity : int
        Neighborhood connectivity to evaluate. Valid values are 4 or 8.
        Default value = 4.

    Returns:
    --------
    X : array_like
        A list of 1D arrays of horizontal coordinates for each object in
        'Label'.
    Y : array_like
        A list of 1D arrays of horizontal coordinates for each object in
        'Label'.

    Notes:
    ------
        Objects should be made contiguous using SplitLabel prior to tracing.
        Returns lists with length Label.max(). Object values missing from
    'Label' will have corresponding boundaries with value 'None' in the
    outputs. Condensing the label image values prior to boundary tracing can
    prevent this.
        Uses the Improved Simple Boundary Follower (ISBF) from the reference
    below for 4-connected tracing. This algorithm provides accurate tracing
    with competitive execution times. 8-connected tracing is implemented using
    the Moore tracing algorithm.

    See Also:
    ---------
    SplitLabel, CondenseLabel, TraceBounds

    References:
    -----------
    .. [1] J. Seo et al "Fast Contour-Tracing Algorithm Based on a Pixel-
    Following Method for Image Sensors" in Sensors,vol.16,no.353,
    doi:10.3390/s16030353, 2016.
    """

    # initialize list of lists containing contours
    X = []
    Y = []

    # get extent of each object
    Locations = ms.find_objects(Label)

    # process each seed pixel sequentially
    for i in np.arange(1, Label.max()+1):

        # process non-empty objects
        if not Locations[i] is None:

            # capture patch containing object of interest
            Patch = Label[Locations[i]] == i

            # pad edges for tracing
            Embed = np.zeros((Patch.shape[0]+2, Patch.shape[1]+2),
                             dtype=np.bool)
            Embed[1:-1, 1:-1] = Patch

            # trace boundary
            cX, cY = lb.TraceBounds(Embed, Connectivity)

            # add window offset to contour coordinates
            cX = cX + Locations[i][1].start - 1
            cY = cY + Locations[i][0].start - 1

            # append to list of candidate contours
            X.append(np.array(cX, dtype=np.uint32))
            Y.append(np.array(cY, dtype=np.uint32))

    else:

        # append None to Outputs X, Y
        X.append(None)
        Y.append(None)

    return X, Y
