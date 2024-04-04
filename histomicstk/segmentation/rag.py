import numpy as np


def rag(im_label, neigh_conn=4):
    """Constructs a region adjacency graph for a label image using either
    4-neighbor or 8-neighbor connectivity. Background pixels are not included
    (im_label == 0). Not intended to build large graphs from individual pixels.

    Parameters
    ----------
    im_label : array_like
        im_label image where positive values (im_label > 0) correspond to foreground
        objects of interest.

    neigh_conn : float
        The neighbor connectivity to use, either '4' or '8'. Default value = 4.

    Returns
    -------
    adj_mat : array_like
        A binary matrix of size N x N, where N is the number of objects in
        im_label. A value of 'True' at adj_mat(i,j) indicates that objects 'i'
        and 'j' are neigh_conn.

    """
    # initialize adjacency matrix for im_label.max() total regions
    adj_mat = np.zeros((im_label.max(), im_label.max()), dtype=bool)

    # process 4-neigh_conn horizontal connections
    Mask = (im_label[:, 1:-1] != im_label[:, 0:-2]) & \
           (im_label[:, 1:-1] != 0) & (im_label[:, 0:-2] != 0)
    Indices = np.nonzero(Mask)
    Xc = np.concatenate((im_label[Indices[0], Indices[1]][:, np.newaxis],
                         im_label[Indices[0], Indices[1] + 1][:, np.newaxis]),
                        axis=1)

    # process 4-neigh_conn vertical connections
    Mask = (im_label[1:-1, :] != im_label[0:-2, :]) & \
           (im_label[1:-1, :] != 0) & (im_label[0:-2, :] != 0)
    Indices = np.nonzero(Mask)
    Xc = np.concatenate((Xc,
                         np.concatenate((im_label[Indices[0], Indices[1]]
                                         [:, np.newaxis],
                                         im_label[Indices[0] + 1, Indices[1]]
                                         [:, np.newaxis]), axis=1)),
                        axis=0)

    # process additional 8-neighbor relationships
    if neigh_conn == 8:

        # shift upper-right ([1, 1])
        Mask = (im_label[1:-1, 0:-2] != im_label[0:-2, 1:-1]) & \
               (im_label[1:-1, 0:-2] != 0) & (im_label[0:-2, 1:-1] != 0)
        Indices = np.nonzero(Mask)
        Xc = np.concatenate((
            Xc,
            np.concatenate((
                im_label[Indices[0] + 1, Indices[1]][:, np.newaxis],
                im_label[Indices[0], Indices[1] + 1][:, np.newaxis]),
                axis=1)),
            axis=0,
        )

        # shift upper-left ([-1, 1])
        Mask = (im_label[0:-2, 0:-2] != im_label[1:-1, 1:-1]) & \
               (im_label[0:-2, 0:-2] != 0) & (im_label[1:-1, 1:-1] != 0)
        Indices = np.nonzero(Mask)
        Xc = np.concatenate((
            Xc,
            np.concatenate((
                im_label[Indices[0], Indices[1]][:, np.newaxis],
                im_label[Indices[0] + 1, Indices[1] + 1][:, np.newaxis]),
                axis=1)),
            axis=0,
        )

    # add entries to adjacency matrix
    for i in range(Xc.shape[0]):
        adj_mat[Xc[i, 0] - 1, Xc[i, 1] - 1] = True
        adj_mat[Xc[i, 1] - 1, Xc[i, 0] - 1] = True

    # return result
    return adj_mat
