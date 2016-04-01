import numpy as np
import skimage.morphology as mp
from skimage import measure as ms


def MergeSinks(Label, Sinks, Radius=5):
    """
    Merges attraction basins obtained from gradient flow tracking using
    sink locations.

    Parameters
    ----------
    Segmentation : array_like
        Label image where positive values correspond to foreground pixels that
        share mutual sinks.
    Sinks : array_like
        N x 2 array containing the (x,y) locations of the tracking sinks. Each
        row is an (x,y) pair - in that order.
    Radius : float
        Radius used to merge sinks. Sinks closer than this radius to one
        another will have their regions of attraction merged.
        Default value = 5.

    Returns
    -------
    Merged : array_like
        Label image where attraction regions are merged.

    See Also
    --------
    GradientDiffusion, MergeSeeds, ShuffleLabel
    """

    # build seed image
    SeedImage = np.zeros(Label.shape)
    for i in range(Sinks.shape[0]):
        SeedImage[Sinks[i, 1], Sinks[i, 0]] = i+1

    # dilate sink image
    Dilated = mp.binary_dilation(SeedImage, mp.disk(Radius))

    # generate new labels for merged seeds, define memberships
    Labels = ms.label(Dilated)
    New = Labels[Sinks[:, 1].astype(np.int), Sinks[:, 0].astype(np.int)]

    # get unique list of seed clusters
    Unique = np.arange(1, New.max()+1)

    # generate new seed list
    Merged = np.zeros(Label.shape)

    # get pixel list for each sink object
    Props = ms.regionprops(Label.astype(np.int))

    # fill in new values
    for i in Unique:
        Indices = np.nonzero(New == i)[0]
        for j in Indices:
            Coords = Props[j].coords
            Merged[Coords[:, 0], Coords[:, 1]] = i

    return Merged
