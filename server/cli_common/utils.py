import numpy

from histomicstk.preprocessing.color_deconvolution import stain_color_map


def get_stain_vector(args, index):
    """Get the stain corresponding to args.stain_$index and
    args.stain_$index_vector.  If the former is not "custom", the
    latter must be None.

    """
    args = vars(args)
    stain = args['stain_' + str(index)]
    stain_vector = args['stain_' + str(index) + '_vector']
    if stain_vector is None:
        if stain == 'custom':
            raise ValueError('If "custom" is chosen for a stain, '
                             'a stain vector must be provided.')
        return stain_color_map[stain]
    else:
        if stain == 'custom':
            return stain_vector
        raise ValueError('Unless "custom" is chosen for a stain, '
                         'no stain vector may be provided.')


def get_stain_matrix(args, count=3):
    """Get the stain matrix corresponding to the args.stain_$index and
    args.stain_$index_vector arguments for values of index 1 to count.
    Return a numpy array of column vectors.

    """
    return numpy.array([get_stain_vector(args, i+1) for i in range(count)]).T


__all__ = (
    'get_stain_vector',
    'get_stain_matrix',
)
