def convert_image_to_matrix(im):
    """Convert an image (MxNx3 array) to a column matrix of pixels
    (3x(M*N)).  It will pass through a 2D array unchanged.

    """
    if im.ndim == 2:
        return im

    return im.reshape((-1, im.shape[-1])).T
