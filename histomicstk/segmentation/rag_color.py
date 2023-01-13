import numpy as np


def rag_color(adj_mat):
    """Generates a coloring of an adjacency graph using the sequential coloring
    algorithm. Used to bin regions from a label image into a small number of
    independent groups that can be processed separately with algorithms like
    multi-label graph cuts or individual active contours. The rationale is to
    color adjacent objects with distinct colors so that their contours can be
    co-evolved.

    Parameters
    ----------
    adj_mat : array_like
        A binary matrix of size N x N, where N is the number of objects in
        Label. A value of 'True' at adj_mat(i,j) indicates that objects 'i'
        and 'j' are neighbors. Does not contain entries for background objects.

    Returns
    -------
    Colors : array_like
        A list of colors for the objects encoded in 'adj_mat'. No two objects
        that are connected in 'adj_mat' will share the same color.

    See Also
    --------
    histomicstk.segmentation.rag,
    histomicstk.segmentation.rag_add_layer

    """

    # initialize colors and color count
    Colors = np.zeros((adj_mat.shape[0], 1), dtype=int)
    Colors[0] = 1
    ColorCount = 1

    # iterate over remaining nodes in order, finding legal coloring
    for i in range(1, adj_mat.shape[0]):

        # get indices neighbors of node 'i'
        Neighbors = np.nonzero(adj_mat[i, ])[0].flatten()

        if Neighbors.size > 0:

            # get colors of neighbors
            NeighborColors = Colors[Neighbors]
            NeighborColors = NeighborColors[np.nonzero(NeighborColors)]

            # check if neighbors have been labeled
            if NeighborColors.size > 0:

                # find lowest legal color of node 'i'
                Reference = set(range(1, ColorCount + 1))
                Diff = Reference.difference(set(NeighborColors))
                if len(Diff) == 0:
                    ColorCount += 1
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
