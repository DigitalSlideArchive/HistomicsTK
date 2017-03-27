from histomicstk.preprocessing import color_conversion
from histomicstk.utils.linalg import normalize
from .complement_stain_matrix import complement_stain_matrix
import collections
import numpy


def color_deconvolution(im_rgb, w):
    """Performs color deconvolution.
    The given RGB Image `I` is first first transformed into optical density
    space, and then projected onto the stain vectors in the columns of the
    3x3 stain matrix `W`.

    For deconvolving H&E stained image use:

    `w` = array([[0.650, 0.072, 0], [0.704, 0.990, 0], [0.286, 0.105, 0]])

    Parameters
    ----------
    im_rgb : array_like
        Input RGB Image that needs to be deconvolved.
    w : array_like
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
        with color_convolution.
    Wc : array_like
        A 3x3 complemented stain matrix. Useful for color image
        reconstruction with color_convolution.

    See Also
    --------
    histomicstk.preprocessing.color_deconvolution.complement_stain_matrix,
    histomicstk.preprocessing.color_deconvolution.color_convolution
    histomicstk.preprocessing.color_conversion.rgb_to_od
    histomicstk.preprocessing.color_conversion.od_to_rgb
    """

    # complement stain matrix if needed
    if numpy.linalg.norm(w[:, 2]) <= 1e-16:
        wc = complement_stain_matrix(w)
    else:
        wc = w

    # normalize stains to unit-norm
    wc = normalize(wc)

    # invert stain matrix
    Q = numpy.linalg.inv(wc)

    # transform 3D input image to 2D RGB matrix format
    m = im_rgb.shape[0]
    n = im_rgb.shape[1]
    if im_rgb.shape[2] == 4:
        im_rgb = im_rgb[:, :, (0, 1, 2)]
    im_rgb = numpy.reshape(im_rgb, (m * n, 3))

    # transform input RGB to optical density values and deconvolve,
    # tfm back to RGB
    im_rgb = im_rgb.astype(dtype=numpy.float32)
    im_rgb[im_rgb == 0] = 1e-16
    ODfwd = color_conversion.rgb_to_od(im_rgb)
    ODdeconv = numpy.dot(ODfwd, numpy.transpose(Q))
    ODinv = color_conversion.od_to_rgb(ODdeconv)

    # reshape output
    StainsFloat = numpy.reshape(ODinv, (m, n, 3))

    # transform type
    Stains = numpy.copy(StainsFloat)
    Stains[Stains > 255] = 255
    Stains = Stains.astype(numpy.uint8)

    # return
    Unmixed = collections.namedtuple('Unmixed',
                                     ['Stains', 'StainsFloat', 'Wc'])
    Output = Unmixed(Stains, StainsFloat, wc)

    return Output
