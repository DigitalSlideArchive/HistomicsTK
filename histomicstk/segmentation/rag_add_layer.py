import numpy as np


def rag_add_layer(adj_mat):
    """Adds an additional layer of dependence to a region adjacency graph,
    connecting each node to the neighbors of its immediate neighbors.

    Parameters
    ----------
    adj_mat : array_like
        A binary matrix of size N x N, where N is the number of objects in
        Label. A value of 'True' at adj_mat(i,j) indicates that objects 'i'
        and 'j' are neighbors.

    Returns
    -------
    Layered : array_like
        A version of 'adj_mat' with additional edges to connect 2-neighbors.

    See Also
    --------
    histomicstk.segmentation.rag

    """
    # initialize output
    Layered = adj_mat.copy()

    # iterate through each object, adding edges to neighbors of neighbors
    for i in range(adj_mat.shape[0]):

        # get immediate neighbors
        Neighbors = np.nonzero(adj_mat[i, ])[0].flatten()

        # for each immediate neighbor, add second neighbors
        for j in range(Neighbors.size):
            Hops = np.nonzero(adj_mat[Neighbors[j], ])[0].flatten()
            Layered[Hops, i] = True
            Layered[i, Hops] = True

    # return result
    return Layered
