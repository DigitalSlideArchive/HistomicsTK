import numpy as np


def fit_poisson_mixture(im_input, mu=None, tol=0.1):
    """Generates a Poisson mixture model to fit pixel intensities for
    foreground/background masking.

    Takes as input an array or intensity image 'I' and optimizes a
    two-component poisson model describing foreground and background intensity
    models. This model can be used to describe the probability that a pixel
    comes from foreground versus background. The poisson distribution assumes
    discrete values and so is suitable for integral valued intensity images.
    Assumes that foreground intensities are lower (darker) than background.

    Parameters
    ----------
    im_input : array_like
        A hematoxylin intensity image obtained from ColorDeconvolution.
    mu : double
        Optional mean value of signal to optimize. Calculated from input if
        defined as 'None'. Default value = None.

    Returns
    -------
    thresh : double
        Optimal threshold for distinguishing foreground and background.
    im_fgnd : array_like
        An intensity image with values in the range [0, 1] representing
        foreground probabiities for each pixel.
    im_bgnd : array_like
        An intensity image with values in the range [0, 1] representing
        background probabiities for each pixel.

    References
    ----------
    .. [#] Y. Al-Kofahi et al "Improved Automatic Detection and Segmentation
       of Cell Nuclei in Histopathology Images" in IEEE Transactions on
       Biomedical Engineering,vol.57,no.4,pp.847-52, 2010.

    """
    import scipy as sp

    # check if intensity values in 'I' are integer type
    if not np.issubdtype(im_input.dtype, np.integer):
        msg = (
            'Inputs for Poisson mixture modeling should be integer'
            ' type'
        )
        raise TypeError(msg)

    # generate a small number for conditioning calculations
    Small = np.finfo(float).eps

    # generate histogram of inputs - assume range is 0, 255 (type uint8)
    H = np.histogram(np.ravel(im_input), bins=256, range=(0, 256))
    X = H[1]
    H = H[0].astype('float') / H[0].sum()

    if mu is None:
        mu = np.dot(X[0: -1], H)

    # calculate cumulative sum along histogram counts
    Cumulative = np.cumsum(H)
    CumProd = np.cumsum(np.multiply(X[0: -1], H))

    # determine cost at each possible threshold
    P0 = Cumulative
    P0[P0 <= 0] = Small
    P1 = 1 - Cumulative
    P1[P1 <= 0] = Small
    Mu0 = np.divide(CumProd, P0) + Small
    Mu1 = np.divide(CumProd[-1] - CumProd, P1) + Small
    Cost = mu - np.multiply(P0, np.log(P0) + np.multiply(Mu0, np.log(Mu0))) - \
        np.multiply(P1, np.log(P1) + np.multiply(Mu1, np.log(Mu1)))

    # identify minimum cost threshold
    thresh = X[np.argmin(Cost)]
    Mu0 = Mu0[np.argmin(Cost)]
    Mu1 = Mu1[np.argmin(Cost)]

    # build probability distribution of foreground intensity values
    Poisson = sp.stats.poisson(Mu0)
    Model = Poisson.pmf(np.arange(0, 256))
    im_fgnd = Model[im_input]
    Poisson = sp.stats.poisson(Mu1)
    Model = Poisson.pmf(np.arange(0, 256))
    im_bgnd = Model[im_input]

    # return threshold and foreground probabilities
    return thresh, im_fgnd, im_bgnd
