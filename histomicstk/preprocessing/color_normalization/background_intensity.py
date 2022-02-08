import numpy

from histomicstk.utils import sample_pixels


def background_intensity(*args, **kwargs):
    """Sample the background of the slide identified by slide_path to
    compute the background intensities in the RGB channels.  Arguments
    are as in histomicstk.utils.sample_pixels, with background forced
    to True.

    Notes
    -----
    The `magnification` parameter defaults to 1.25x, instead of the
    native scan magnification as in sample_pixels.

    """
    if 'background' in kwargs:
        raise ValueError('"background" argument may not be specified')

    kwargs['background'] = True

    if 'magnification' not in kwargs:
        kwargs['magnification'] = 1.25

    sample = sample_pixels(*args, **kwargs)

    return numpy.median(sample, axis=0)
