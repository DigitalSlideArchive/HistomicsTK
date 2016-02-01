import numpy as np

from . import OpticalDensityFwd
from . import OpticalDensityInv


def ColorConvolution(I, W):
    """Performs Color Convolution
    Reconstructs a color image from the stain matrix `W` and
    the individual images stored as channels in `I` and generated
    by ColorDeconvolution.

    Parameters
    ----------
    I : array_like
        An RGB image where in each channel contains image of one stain
    W : array_like
        A 3x3 matrix containing the stain colors in its columns.
        In the case of two stains, the third column is zero and will be
        complemented using cross-product. The matrix should contain a
        minumum two nonzero columns.

    Returns
    -------
    IOut : array_like
        Reconstructed RGB image with intensity values ranging from [0, 255],
        suitable for display.

    See Also
    --------
    ComplementStainMatrix, OpticalDensityFwd
    OpticalDensityInv, ColorDeconvolution
    """

    # transform 3D input stain image to 2D stain matrix format
    m = I.shape[0]
    n = I.shape[1]
    I = np.reshape(I, (m*n, 3))

    # transform input stains to optical density values, convolve and
    # tfm back to stain
    I = I.astype(dtype=np.float32)
    ODfwd = OpticalDensityFwd(I)
    ODdeconv = np.dot(ODfwd, np.transpose(W))
    ODinv = OpticalDensityInv(ODdeconv)

    # reshape output, transform type
    IOut = np.reshape(ODinv, (m, n, 3))
    IOut[IOut > 255] = 255
    IOut = IOut.astype(np.uint8)
    
    return(IOut)
