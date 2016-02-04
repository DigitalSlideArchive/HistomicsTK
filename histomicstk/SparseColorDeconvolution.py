import nimfa
import numpy as np
import collections

from OpticalDensityFwd import OpticalDensityFwd


def SparseColorDeconvolution(I, Winit, Beta):
    """Performs adaptive color deconvolution.

    Uses sparse non-negative matrix factorization to adaptively deconvolve a
    given RGB image into intensity images representing distinct stains.
    Similar approach to ``ColorDeconvolution`` but operates adaptively.
    The input RGB image `I` consisting of RGB values is first transformed
    into optical density space as a row-matrix, and then is decomposed as
    :math:`V = W H` where :math:`W` is a 3xk matrix containing stain vectors
    in columns and :math:`H` is a k x m*n matrix of concentrations for each
    stain vector. The system is solved to encourage sparsity of the columns
    of :math"`H` i.e. most pixels should not contain significant contributions
    from more than one stain. Can use a hot-start initialization from a color
    deconvolution matrix.

    Parameters
    ----------
    I : array_like
        An RGB image of type unsigned char, or a 3xN matrix of RGB pixel
        values.
    Winit : array_like
        A 3xK matrix containing the color vectors in columns. Should not be
        complemented with ComplementStainMatrix for sparse decomposition to
        work correctly.
    Beta : double
        Regularization factor for sparsity of :math:`H` - recommended 0.5.

    Returns
    -------
    Stains : array_like
        An rgb image with deconvolved stain intensities in each channel,
        values ranging from [0, 255], suitable for display.
    W : array_like
        The final 3 x k stain matrix produced by NMF decomposition.

    Notes
    -----
    Return values are returned as a namedtuple

    See Also
    --------
    ColorDeconvolution

        References
    ----------
    .. [1] J. Xu, L. Xiang, G. Wang, S. Ganesan, M. Feldman, N.N. Shih,
           H. Gilmore, A. Madabhushi, "Sparse Non-negative Matrix
           Factorization (SNMF) based color unmixing for breast
           histopathological image analysis," in IEEE Computer Graphics
           and Applications, vol.46,no.1,pp.20-9, 2015.
    """

    # determine if input is RGB or pixel-matrix format
    if len(I.shape) == 3:  # RBG image provided
        m = I.shape[0]
        n = I.shape[1]
        I = np.reshape(I, (m * n, 3)).transpose()
    elif len(I.shape) == 2:  # pixel matrix provided
        m = -1
        n = -1
        if I.shape[2] == 4:  # remove alpha channel if needed
            I = I[:, :, (0, 1, 2)]

    # transform input RGB to optical density values
    I = I.astype(dtype=np.float32)
    I[I == 0] = 1e-16
    ODfwd = OpticalDensityFwd(I)

    if Winit is None:

        # set number of output stains
        K = 3

        # perform NMF without initialization
        Factorization = nimfa.Snmf(V=ODfwd, seed=None, rank=K,
                                   version='r', beta=Beta)
        Factorization()

    else:

        # get number of output stains
        K = Winit.shape[1]

        # normalize stains to unit-norm
        for i in range(K):
            Norm = np.linalg.norm(Winit[:, i])
            if(Norm >= 1e-16):
                Winit[:, i] /= Norm
            else:
                print 'error'  # throw error

        # estimate initial H given p
        Hinit = np.dot(np.linalg.pinv(Winit), ODfwd)
        Hinit[Hinit < 0] = 0

        # perform regularized NMF
        Factorization = nimfa.Snmf(V=ODfwd, seed=None, W=Winit,
                                   H=Hinit, rank=K,
                                   version='r', beta=Beta)
        Factorization()

    # extract solutions and make columns of "W" unit-norm
    W = np.asarray(Factorization.basis())
    H = np.asarray(Factorization.coef())
    for i in range(K):
        Norm = np.linalg.norm(W[:, i])
        W[:, i] /= Norm
        H[i, :] *= Norm

    # reshape H matrix to image
    if m == -1:
        StainsFloat = np.transpose(H)
    else:
        StainsFloat = np.reshape(np.transpose(H), (m, n, K))

    # transform type
    Stains = np.copy(StainsFloat)
    Stains[Stains > 255] = 255
    Stains = Stains.astype(np.uint8)

    # build named tuple for outputs
    Unmixed = collections.namedtuple('Unmixed', ['Stains', 'W'])
    Output = Unmixed(Stains, W)

    # return solution
    return Output
