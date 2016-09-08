import numpy as np
import scipy.ndimage.filters as ft
import scipy.ndimage.morphology as mp


def CompactLabel(Label, Compaction=3):
    """Performs a thinning operation on a label image to remove thin
    protrusions from objects that are in contact with the background. Applies a
    distance transform and sequentially removes pixels with small distances to
    background that are not connected to higher distance pixels.

    Parameters
    ----------
    Label : array_like
        A labeled segmentation mask
    Compaction : int
        Factor used in compacting objects to remove thin protrusions. Refered
        to as d in the reference below. Default value = 3.

    Notes
    -----
    Implemented from the reference below.

    Returns
    -------
    Compact : array_like
        A labeled segmentation mask with thin protrusions removed.

    See Also
    --------
    histomicstk.segmentation.label.AreaOpenLabel,
    histomicstk.segmentation.label.CondenseLabel,
    histomicstk.segmentation.label.ShuffleLabel,
    histomicstk.segmentation.label.SplitLabel,
    histomicstk.segmentation.label.WidthOpenLabel

    References
    ----------
    .. [1] S. Weinert et al "Detection and Segmentation of Cell Nuclei in
           Virtual Microscopy Images: A Minimum-Model Approach" in Nature
           Scientific Reports,vol.2,no.503, doi:10.1038/srep00503, 2012.
    """

    # copy input image
    Compact = Label.copy()

    # generate distance map of label image
    D = mp.distance_transform_cdt(Compact > 0, metric='taxicab')

    # define 4-neighbors filtering kernel
    Kernel = np.zeros((3, 3), dtype=np.bool)
    Kernel[1, :] = True
    Kernel[:, 1] = True

    # sweep over distance values from d-1 to 1
    for i in np.arange(Compaction-1, 0, -1):

        # four-neighbor maxima of distance transform
        MaxD = ft.maximum_filter(D, footprint=Kernel)

        # identify pixels whose max 4-neighbor is less than i+1
        Decrement = (D == i) & (MaxD < i+1)

        # decrement non-compact pixels
        D[Decrement] -= 1

    # zero label pixels where D == 0
    Compact[D == 0] = 0

    return Compact
