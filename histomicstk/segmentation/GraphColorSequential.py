import numpy as np
from sets import Set


def GraphColorSequential(Adjacency):
    """Generates a coloring of an adjacency graph using the sequential coloring
    algorithm. Used to bin regions from a label image into a small number of
    independent groups that can be processed seperately with algorithms like
    multilabel graph cuts or individual active contours. The rationale is to
    color adjacent objects with distinct colors so that their contours can be
    co-evolved.

    Parameters
    ----------
    Adjacency : array_like
        A binary matrix of size N x N, where N is the number of objects in
        Label. A value of 'True' at Adjacency(i,j) indicates that objects 'i'
        and 'j' are neighbors. Does not contain entries for background objects.

    Returns
    -------
    Colors : array_like
        A list of colors for the objects encoded in 'Adjacency'. No two objects
        that are connected in 'Adjacency' will share the same color.

    See Also
    --------
    LabelRegionAdjacency, RegionAdjacencyLayer, GraphcutRefine
    """

    # initialize colors and color count
    Colors = np.zeros((Adjacency.shape[0], 1), dtype=np.int)
    Colors[0] = 1
    ColorCount = 1

    # iterate over remaining nodes in order, finding legal coloring
    for i in range(1, Adjacency.shape[0]):

        # get indices neighbors of node 'i'
        Neighbors = np.nonzero(Adjacency[i, ])[0].flatten()
        if(Neighbors.size > 0):

            # get colors of neighbors
            NeighborColors = Colors[Neighbors]
            NeighborColors = NeighborColors[np.nonzero(NeighborColors)]

            # check if neighbors have been labeled
            if NeighborColors.size > 0:

                # find lowest legal color of node 'i'
                Reference = Set(range(1, ColorCount+1))
                Diff = Reference.difference(Set(NeighborColors))
                if len(Diff) == 0:
                    ColorCount = ColorCount + 1
                    Colors[i] = ColorCount
                else:
                    Colors[i] = min(Diff)
            else:

                # no other neighbors have been labeled yet - set value = 1
                Colors[i] = 1

        else:  # object is an island - no neighbors

            # set to base color
            Colors[i] = 1

    return Colors.flatten()
