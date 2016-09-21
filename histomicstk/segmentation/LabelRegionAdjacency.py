import numpy as np


def LabelRegionAdjacency(Label, Neighbors=4):
    """Constructs a region adjacency graph for a label image using either
    4-neighbor or 8-neighbor connectivity. Background pixels are not included
    (Label == 0). Not intended to build large graphs from individual pixels.

    Parameters
    ----------
    Label : array_like
        Label image where positive values (Label > 0) correspond to foreground
        objects of interest.

    Neighbors : float
        The neighbor connectivity to use, either '4' or '8'. Default value = 4.

    Returns
    -------
    Adjacency : array_like
        A binary matrix of size N x N, where N is the number of objects in
        Label. A value of 'True' at Adjacency(i,j) indicates that objects 'i'
        and 'j' are neighbors.
    """

    # initialize adjacency matrix for Label.max() total regions
    Adjacency = np.zeros((Label.max(), Label.max()), dtype=np.bool)

    # process 4-neighbors horizontal connections
    Mask = (Label[:, 1:-1] != Label[:, 0:-2]) & \
           (Label[:, 1:-1] != 0) & (Label[:, 0:-2] != 0)
    Indices = np.nonzero(Mask)
    Xc = np.concatenate((Label[Indices[0], Indices[1]][:, np.newaxis],
                         Label[Indices[0], Indices[1]+1][:, np.newaxis]),
                        axis=1)

    # process 4-neighbors vertical connections
    Mask = (Label[1:-1, :] != Label[0:-2, :]) & \
           (Label[1:-1, :] != 0) & (Label[0:-2, :] != 0)
    Indices = np.nonzero(Mask)
    Xc = np.concatenate((Xc,
                         np.concatenate((Label[Indices[0], Indices[1]]
                                         [:, np.newaxis],
                                         Label[Indices[0]+1, Indices[1]]
                                         [:, np.newaxis]), axis=1)),
                        axis=0)

    # process additional 8-neighbor relationships
    if Neighbors == 8:

        # shift upper-right ([1, 1])
        Mask = (Label[1:-1, 0:-2] != Label[0:-2, 1:-1]) & \
               (Label[1:-1, 0:-2] != 0) & (Label[0:-2, 1:-1] != 0)
        Indices = np.nonzero(Mask)
        Xc = np.concatenate((Xc,
                             np.concatenate((Label[Indices[0]+1, Indices[1]]
                                             [:, np.newaxis],
                                             Label[Indices[0], Indices[1]+1]
                                             [:, np.newaxis]), axis=1)),
                            axis=0)

        # shift upper-left ([-1, 1])
        Mask = (Label[0:-2, 0:-2] != Label[1:-1, 1:-1]) & \
               (Label[0:-2, 0:-2] != 0) & (Label[1:-1, 1:-1] != 0)
        Indices = np.nonzero(Mask)
        Xc = np.concatenate((Xc,
                             np.concatenate((Label[Indices[0], Indices[1]]
                                             [:, np.newaxis],
                                             Label[Indices[0]+1, Indices[1]+1]
                                             [:, np.newaxis]), axis=1)),
                            axis=0)

    # add entries to adjacency matrix
    for i in range(Xc.shape[0]):
        Adjacency[Xc[i, 0]-1, Xc[i, 1]-1] = True
        Adjacency[Xc[i, 1]-1, Xc[i, 0]-1] = True

    # return result
    return Adjacency
