import numpy as np


def RegionAdjacencyLayer(Adjacency):
    """Adds an additional layer of dependence to a region adjacency graph,
    connecting each node to the neighbors of its immediate neighbors.

    Parameters
    ----------
    Adjacency : array_like
        A binary matrix of size N x N, where N is the number of objects in
        Label. A value of 'True' at Adjacency(i,j) indicates that objects 'i'
        and 'j' are neighbors.

    Returns
    -------
    Layered : array_like
        A version of 'Adjacency' with additional edges to connect 2-neighbors.

    See Also
    --------
    LabelRegionAdjacency, GraphcutRefine
    """

    # initialize output
    Layered = Adjacency.copy()

    # iterate through each object, adding edges to neighbors of neighbors
    for i in range(Adjacency.shape[0]):

        # get immediate neighbors
        Neighbors = np.nonzero(Adjacency[i, ])[0].flatten()

        # for each immediate neighbor, add second neighbors
        for j in range(Neighbors.size):
            Hops = np.nonzero(Adjacency[Neighbors[j], ])[0].flatten()
            Layered[Hops, i] = True
            Layered[i, Hops] = True

    # return result
    return Layered
