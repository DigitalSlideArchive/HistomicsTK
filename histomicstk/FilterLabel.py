from CondenseLabel import CondenseLabel
import numpy as np
from skimage import measure as ms


def FilterLabel(Label, Neighbors=4, Lower=100, Upper=None):
    """Filter small objects from a label image and compress label values to
    fill gaps.

    Parameters
    ----------
    Label : array_like
        Label image where positive values correspond to foreground pixels that
        share mutual sinks.
    Neighbors : float
        The nieghbor connectivity to use, either '4' or '8'. Default value = 4.
    Lower : float
        Minimum area threshold (pixels) under which objects will be discarded.
        Default value = 100.
    Upper : float
        Maximum area threshold (pixels) over which objects will be discarded.
        Default value = None.

    Returns
    -------
    Filtered : array_like
        A label image where small objects are discarded and values > 0 are
        shifted down to fill gaps.

    See Also
    --------
    CondenseLabel
    """

    # copy input to output
    Filtered = Label.copy()

    # re-label and get areas associated with label values
    NewLabel = ms.label(Label > 0, neighbors=Neighbors, background=0)
    Properties = ms.regionprops(NewLabel)
    Areas = np.zeros(NewLabel.max(), dtype=np.int)
    for i, region in enumerate(Properties):
        Areas[i] = region.area

    # identify indices of objects to delete
    Deleted = np.zeros(0, dtype=np.int)
    if Lower is not None:
        Deleted = np.nonzero(Areas < Lower)[0]
    if Upper is not None:
        Deleted = np.concatenate((Deleted,
                                 np.nonzero(Areas > Upper)[0]))

    # zero objects in delete list
    for i in Deleted:
        Filtered[NewLabel == (i+1)] = 0

    # condense label values
    Filtered = CondenseLabel(Filtered)

    return Filtered
