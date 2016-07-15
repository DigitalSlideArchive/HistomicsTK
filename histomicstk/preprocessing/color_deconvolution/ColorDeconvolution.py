from histomicstk.preprocessing import color_conversion
from histomicstk.preprocessing import color_deconvolution

import collections
import numpy


def ColorDeconvolution(I, W):
    """Performs color deconvolution.
    The given RGB Image `I` is first first transformed into optical density
    space, and then projected onto the stain vectors in the columns of the
    3x3 stain matrix `W`.

    For deconvolving H&E stained image use:

    `W` = array([[0.650, 0.072, 0], [0.704, 0.990, 0], [0.286, 0.105, 0]])

    Parameters
    ----------
    I : array_like
        Input RGB Image that needs to be deconvolved.
    W : array_like
        A 3x3 matrix containing the color vectors in columns.
        For two stain images the third column is zero and will be
        complemented using cross-product. Atleast two of the three
        columns must be non-zero.

    Returns
    -------
    Stains : array_like
        An rgb image where in each channel contains the image of the
        stain of the corresponding column in the stain matrix `W`.
        The intensity range of each channel is [0, 255] suitable for
        displaying.
    StainsFloat : array_like
        An intensity image of deconvolved stains that is unbounded,
        suitable for reconstructing color images of deconvolved stains
        with ColorConvolution.
    Wc : array_like
        A 3x3 complemented stain matrix. Useful for color image
        reconstruction with ColorConvolution.

    See Also
    --------
    histomicstk.preprocessing.color_deconvolution.ComplementStainMatrix,
    histomicstk.preprocessing.color_deconvolution.ColorConvolution
    histomicstk.preprocessing.color_conversion,OpticalDensityFwd
    histomicstk.preprocessing.color_conversion,OpticalDensityInv
    """

    # complement stain matrix if needed
    if numpy.linalg.norm(W[:, 2]) <= 1e-16:
        Wc = color_deconvolution.ComplementStainMatrix(W)
    else:
        Wc = W.copy()

    # normalize stains to unit-norm
    for i in range(Wc.shape[1]):
        Norm = numpy.linalg.norm(Wc[:, i])
        if Norm >= 1e-16:
            Wc[:, i] /= Norm

    # invert stain matrix
    Q = numpy.linalg.inv(Wc)

    # transform 3D input image to 2D RGB matrix format
    m = I.shape[0]
    n = I.shape[1]
    if I.shape[2] == 4:
        I = I[:, :, (0, 1, 2)]
    I = numpy.reshape(I, (m * n, 3))

    # transform input RGB to optical density values and deconvolve,
    # tfm back to RGB
    I = I.astype(dtype=numpy.float32)
    I[I == 0] = 1e-16
    ODfwd = color_conversion.OpticalDensityFwd(I)
    ODdeconv = numpy.dot(ODfwd, numpy.transpose(Q))
    ODinv = color_conversion.OpticalDensityInv(ODdeconv)

    # reshape output
    StainsFloat = numpy.reshape(ODinv, (m, n, 3))

    # transform type
    Stains = numpy.copy(StainsFloat)
    Stains[Stains > 255] = 255
    Stains = Stains.astype(numpy.uint8)

    # return
    Unmixed = collections.namedtuple('Unmixed',
                                     ['Stains', 'StainsFloat', 'Wc'])
    Output = Unmixed(Stains, StainsFloat, Wc)

    return Output
