import numpy

from .stain_color_map import stain_color_map

def get_stain_vector(args, index):
    """Get the stain corresponding to args.stain_$index and
    args.stain_$index_vector.  If the former is not "custom", the
    latter must be None.

    """
    args = vars(args)
    stain = args['stain_' + str(index)]
    stain_vector = args['stain_' + str(index) + '_vector']
    if stain == 'custom':
        if stain_vector is None:
            raise ValueError('If "custom" is chosen for a stain, '
                             'a stain vector must be provided.')
        return stain_vector
    else:
        if stain_vector is None:
            return stain_color_map[stain]
        raise ValueError('Unless "custom" is chosen for a stain, '
                         'no stain vector may be provided.')

def get_stain_matrix(args):
    """Get the stain matrix corresponding to the args.stain_$index and
    args.stain_$index_vector arguments for values of index 1, 2, 3.
    Return a numpy array of column vectors.

    """
    return numpy.array([get_stain_vector(args, i) for i in 1, 2, 3]).T

__all__ = (
    'get_stain_vector',
)
