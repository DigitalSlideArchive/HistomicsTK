import numpy


def exclude_nonfinite(m):
    """Exclude columns from m that have infinities or nans.  In the
    context of color deconvolution, these occur in conversion from RGB
    to SDA when the source has 0 in a channel.
    """
    return m[:, numpy.isfinite(m).all(axis=0)]
