import numpy as np


def embed_boundaries(im_input, im_perim, color=None):
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
    im_input : array_like
        A color or grayscale image.
    im_perim : array_like
        A binary image where object perimeter pixels have value 1, and
        non-perimeter pixels have value 0.
    color : array_like
        A 1 x 3 array of RGB values in the range [0, 255].

    Returns
    -------
    im_embed : array_like
        A color image of type unsigned char where boundary pixels take
        on the color defined by the RGB-triplet 'color'.

    See Also
    --------
    histomicstk.segmentation.label.perimeter

    """
    color = [255, 0, 0] if color is None else color

    # check for consistent shapes between 'im_input' and 'im_perim'
    if im_input.shape[0:2] != im_perim.shape[0:2]:
        msg = "'im_input' and 'im_perim' must have same shape"
        raise ValueError(msg)

    # determine image type
    if np.issubclass_(im_input.dtype.type, np.float_):
        if im_input.max() < 1.0:
            im_input *= 255
            im_input = im_input.astype(np.uint8)
    elif np.issubclass_(im_input.dtype.type, np.bool_):
        im_input = im_input.astype(np.uint8)
        im_input *= 255

    # determine if image is grayscale or RGB
    if len(im_input.shape) == 3:  # color image
        Red = im_input[:, :, 0].copy()
        Green = im_input[:, :, 1].copy()
        Blue = im_input[:, :, 2].copy()
    elif len(im_input.shape) == 2:  # grayscale image
        Red = im_input.copy()
        Green = im_input.copy()
        Blue = im_input.copy()

    # embed boundaries
    Red[im_perim > 0] = color[0]
    Green[im_perim > 0] = color[1]
    Blue[im_perim > 0] = color[2]

    # generate output image
    im_embed = np.dstack((Red[:, :, np.newaxis],
                          Green[:, :, np.newaxis],
                          Blue[:, :, np.newaxis])).astype(np.uint8)

    # concatenate channels to form output
    return im_embed
