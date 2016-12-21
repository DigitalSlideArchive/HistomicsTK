import numpy as np
from histomicstk.preprocessing import color_conversion


def color_convolution(im_stains, w):
    """Performs Color Convolution
    Reconstructs a color image from the stain matrix `w` and
    the individual images stored as channels in `im_stains` and generated
    by ColorDeconvolution.

    Parameters
    ----------
    im_stains : array_like
        An RGB image where in each channel contains image of one stain
    w : array_like
        A 3x3 matrix containing the stain colors in its columns.
        In the case of two stains, the third column is zero and will be
        complemented using cross-product. The matrix should contain a
        minumum two nonzero columns.

    Returns
    -------
    im_rgb : array_like
        Reconstructed RGB image with intensity values ranging from [0, 255],
        suitable for display.

    See Also
    --------
    histomicstk.preprocessing.color_deconvolution.complement_stain_matrix,
    histomicstk.preprocessing.color_deconvolution.color_deconvolution
    histomicstk.preprocessing.color_conversion.rgb_to_od
    histomicstk.preprocessing.color_conversion.od_to_rgb
    """

    # transform 3D input stain image to 2D stain matrix format
    m = im_stains.shape[0]
    n = im_stains.shape[1]
    im_stains = np.reshape(im_stains, (m * n, 3))

    # transform input stains to optical density values, convolve and
    # tfm back to stain
    im_stains = im_stains.astype(dtype=np.float32)
    ODfwd = color_conversion.rgb_to_od(im_stains)
    ODdeconv = np.dot(ODfwd, np.transpose(w))
    ODinv = color_conversion.od_to_rgb(ODdeconv)

    # reshape output, transform type
    im_rgb = np.reshape(ODinv, (m, n, 3))
    im_rgb[im_rgb > 255] = 255
    im_rgb = im_rgb.astype(np.uint8)

    return im_rgb
