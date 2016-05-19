import numpy as np
import pygco as gc


def BinaryMixtureCut(Foreground, Background, I=None, Sigma=25):
    """Performs a binary graph-cut optimization for region masking, given
    foreground and background probabilities for image and

    Parameters
    ----------
    I : array_like
        An intensity image used for calculating the continuity term. If not
        provided, only the data term from the foreground/background model will
        drive the graph-cut optimization. Default value = None.
    Foreground : array_like
        Intensity image the same size as 'I' with values in the range [0, 1]
        representing probabilities that each pixel is in the foreground class.
    Background : array_like
        Intensity image the same size as 'I' with values in the range [0, 1]
        representing probabilities that each pixel is in the background class.
    Sigma : double
        Parameter controling exponential decay rate of continuity term. Units
        are intensity values. Recommended ~10% of dynamic range of image.
        Defaults for image 'I' with range [0, 255]. Default value = 25.

    Notes
    -----
    The graph cutting algorithm is sensitive to the magnitudes of weights in
    the data variable 'D'. Small probabilities will be magnified by the
    logarithm in formulating this cost, and so we add a small positive value
    to each in order to scale them appropriately.

    Returns
    -------
    Mask : array_like
        A binary mask indicating the foreground regions.

    See Also
    --------
    PoissonMixture

    References
    ----------
    .. [1] Y. Al-Kofahi et al "Improved Automatic Detection and Segmentation
    of Cell Nuclei in Histopathology Images" in IEEE Transactions on Biomedical
    Engineering,vol.57,no.4,pp.847-52, 2010.
    """

    # generate a small number for conditioning calculations
    Small = np.finfo(np.float).eps

    # formulate unary data costs
    D = np.stack((-np.log(Background + Small) / -np.log(Small),
                 -np.log(Foreground + Small) / -np.log(Small)),
                 axis=2)

    # formulate pairwise label costs
    Pairwise = 1 - np.eye(2)

    # formulate vertical and horizontal costs if 'I' is provided
    if I is not None:
        H = np.exp(np.abs(I[:, 0:-1].astype(np.float) -
                   I[:, 1:].astype(np.float)) / (2 * Sigma**2))
        V = np.exp(np.abs(I[0:-1, :].astype(np.float) -
                   I[1:, :].astype(np.float)) / (2 * Sigma**2))

        # cut the graph with edge information from image
        Mask = gc.cut_grid_graph(D, Pairwise, V, H, n_iter=-1,
                                 algorithm='expansion')

    else:
        # cut the graph without edge information from image
        Mask = gc.cut_grid_graph_simple(D, Pairwise, n_iter=-1,
                                        algorithm='expansion')

    # reshape solution to image size and return
    return Mask.reshape(Foreground.shape[0], Foreground.shape[1])
