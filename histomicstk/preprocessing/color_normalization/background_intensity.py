from histomicstk.utils import sample_pixels
import numpy

def background_intensity(*args, **kwargs):
    """Sample the background of the slide identified by slide_path to
    compute the background intensities in the RGB channels.  Arguments
    are as in histomicstk.utils.sample_pixels, with background forced
    to True.

    """
    if 'background' in kwargs:
        raise ValueError('"background" argument may not be specified')

    kwargs['background'] = True

    sample = sample_pixels(*args, **kwargs)

    return numpy.median(sample, axis=0)
