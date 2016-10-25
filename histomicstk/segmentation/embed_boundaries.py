import numpy as np


def embed_boundaries(I, Bounds, Color=[255, 0, 0]):
    """Embeds object boundaries into an RGB color, grayscale or binary
    image, returning a color rendering of the image and object
    boundaries.

    Takes as input a grayscale or color image, a perimeter mask of object
    boundaries, and an RGB triplet, and embeds the object boundaries into
    the input image at the prescribed color. Returns a color RGB image of
    type unsigned char. If the input image is type double, and has pixels
    inside range [0, 1], then it will be scaled to the range [0, 255].
    Otherwise it will be assumed to be in the range of an unsigned char
    image.

    Parameters
    ----------
    I : array_like
        A color or grayscale image.
    Bounds : array_like
        A binary image where object perimeter pixels have value 1, and
        non-perimeter pixels have value 0.
    Color : array_like
        A 1 x 3 array of RGB values in the range [0, 255].

    Returns
    -------
    Iout : array_like
        A color image of type unsigned char where boundary pixels take
        on the color defined by the RGB-triplet 'Color'.

    See Also
    --------
    histomicstk.segmentation.label.perimeter
    """

    # check for consistent shapes between 'I' and 'Bounds'
    if I.shape[0:2] != Bounds.shape[0:2]:
        raise ValueError("Inputs 'I' and 'Bounds' must have same shape")

    # determine image type
    if np.issubclass_(I.dtype.type, np.float_):
        if I.max() < 1.0:
            I = I * 255
            I = I.astype(np.uint8)
    elif np.issubclass_(I.dtype.type, np.bool_):
        I = I.astype(np.uint8)
        I = I * 255

    # determine if image is grayscale or RGB
    if len(I.shape) == 3:  # color image
        Red = I[:, :, 0].copy()
        Green = I[:, :, 1].copy()
        Blue = I[:, :, 2].copy()
    elif len(I.shape) == 2:  # grayscale image
        Red = I.copy()
        Green = I.copy()
        Blue = I.copy()

    # embed boundaries
    Red[Bounds > 0] = Color[0]
    Green[Bounds > 0] = Color[1]
    Blue[Bounds > 0] = Color[2]

    # generate output image
    Iout = np.dstack((Red[:, :, np.newaxis],
                      Green[:, :, np.newaxis],
                      Blue[:, :, np.newaxis])).astype(np.uint8)

    # concatenate channels to form output
    return Iout
