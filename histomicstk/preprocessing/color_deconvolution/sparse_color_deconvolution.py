import collections
import nimfa
import numpy as np
from histomicstk.preprocessing import color_conversion


def sparse_color_deconvolution(im_rgb, w_init, beta):
    """Performs adaptive color deconvolution.

    Uses sparse non-negative matrix factorization to adaptively deconvolve a
    given RGB image into intensity images representing distinct stains.
    Similar approach to ``color_deconvolution`` but operates adaptively.
    The input RGB image `im_rgb` consisting of RGB values is first transformed
    into optical density space as a row-matrix, and then is decomposed as
    :math:`V = W H` where :math:`W` is a 3xk matrix containing stain vectors
    in columns and :math:`H` is a k x m*n matrix of concentrations for each
    stain vector. The system is solved to encourage sparsity of the columns
    of :math"`H` i.e. most pixels should not contain significant contributions
    from more than one stain. Can use a hot-start initialization from a color
    deconvolution matrix.

    Parameters
    ----------
    im_rgb : array_like
        An RGB image of type unsigned char, or a 3xN matrix of RGB pixel
        values.
    w_init : array_like
        A 3xK matrix containing the color vectors in columns. Should not be
        complemented with ComplementStainMatrix for sparse decomposition to
        work correctly.
    beta : double
        Regularization factor for sparsity of :math:`H` - recommended 0.5.

    Returns
    -------
    stains : array_like
        An rgb image with deconvolved stain intensities in each channel,
        values ranging from [0, 255], suitable for display.
    w : array_like
        The final 3 x k stain matrix produced by NMF decomposition.

    Notes
    -----
    Return values are returned as a namedtuple

    See Also
    --------
    histomicstk.preprocessing.color_deconvolution.ColorDeconvolution

    References
    ----------
    .. [1] J. Xu, L. Xiang, G. Wang, S. Ganesan, M. Feldman, N.N. Shih,
           H. Gilmore, A. Madabhushi, "Sparse Non-negative Matrix
           Factorization (SNMF) based color unmixing for breast
           histopathological image analysis," in IEEE Computer Graphics
           and Applications, vol.46,no.1,pp.20-9, 2015.
    """

    # determine if input is RGB or pixel-matrix format
    if len(im_rgb.shape) == 3:  # RBG image provided
        m = im_rgb.shape[0]
        n = im_rgb.shape[1]
        im_rgb = np.reshape(im_rgb, (m * n, 3)).transpose()
    elif len(im_rgb.shape) == 2:  # pixel matrix provided
        m = -1
        n = -1
        if im_rgb.shape[2] == 4:  # remove alpha channel if needed
            im_rgb = im_rgb[:, :, (0, 1, 2)]

    # transform input RGB to optical density values
    im_rgb = im_rgb.astype(dtype=np.float32)
    im_rgb[im_rgb == 0] = 1e-16
    ODfwd = color_conversion.rgb_to_od(im_rgb)

    if w_init is None:

        # set number of output stains
        K = 3

        # perform NMF without initialization
        Factorization = nimfa.Snmf(V=ODfwd, seed=None, rank=K,
                                   version='r', beta=beta)
        Factorization()

    else:

        # get number of output stains
        K = w_init.shape[1]

        # normalize stains to unit-norm
        for i in range(K):
            Norm = np.linalg.norm(w_init[:, i])
            if(Norm >= 1e-16):
                w_init[:, i] /= Norm
            else:
                print 'error'  # throw error

        # estimate initial H given p
        Hinit = np.dot(np.linalg.pinv(w_init), ODfwd)
        Hinit[Hinit < 0] = 0

        # perform regularized NMF
        Factorization = nimfa.Snmf(V=ODfwd, seed=None, W=w_init,
                                   H=Hinit, rank=K,
                                   version='r', beta=beta)
        Factorization()

    # extract solutions and make columns of "w" unit-norm
    w = np.asarray(Factorization.basis())
    H = np.asarray(Factorization.coef())
    for i in range(K):
        Norm = np.linalg.norm(w[:, i])
        w[:, i] /= Norm
        H[i, :] *= Norm

    # reshape H matrix to image
    if m == -1:
        stains_float = np.transpose(H)
    else:
        stains_float = np.reshape(np.transpose(H), (m, n, K))

    # transform type
    stains = np.copy(stains_float)
    stains[stains > 255] = 255
    stains = stains.astype(np.uint8)

    # build named tuple for outputs
    Unmixed = collections.namedtuple('Unmixed', ['Stains', 'W'])
    Output = Unmixed(stains, w)

    # return solution
    return Output
