import numpy

from histomicstk.preprocessing.color_deconvolution import stain_color_map


def get_stain_vector(args, index):
    """Get the stain corresponding to args.stain_$index and
    args.stain_$index_vector.  If the former is not "custom", all the
    latter's elements must be -1.

    """
    args = vars(args)
    stain = args['stain_' + str(index)]
    stain_vector = args['stain_' + str(index) + '_vector']
    if all(x == -1 for x in stain_vector):  # Magic default value
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


def get_region_dict(region, maxRegionSize=None, tilesource=None):
    """Return a dict corresponding to region, checking the region size if
    maxRegionSize is provided.

    The intended use is to be passed via **kwargs, and so either {} is
    returned (for the special region -1,-1,-1,-1) or {'region':
    region_dict}.

    Params
    ------
    region: list
        4 elements -- left, top, width, height -- or all -1, meaning the whole
        slide.
    maxRegionSize: int, optional
        Maximum size permitted of any single dimension
    tilesource: tilesource, optional
        A `large_image` tilesource (or anything with `.sizeX` and `.sizeY`
        properties) that is used to determine the size of the whole slide if
        necessary.  Must be provided if `maxRegionSize` is.

    Returns
    -------
    region_dict: dict
        Either {} (for the special region -1,-1,-1,-1) or
        {'region': region_subdict}

    """
    if len(region) != 4:
        raise ValueError('Exactly four values required for --region')

    useWholeImage = region == [-1] * 4

    if maxRegionSize is not None:
        if tilesource is None:
            raise ValueError('tilesource must be provided if maxRegionSize is')
        if maxRegionSize != -1:
            if useWholeImage:
                size = max(tilesource.sizeX, tilesource.sizeY)
            else:
                size = max(region[-2:])
            if size > maxRegionSize:
                raise ValueError('Requested region is too large!  '
                                 'Please see --maxRegionSize')

    return {} if useWholeImage else dict(
        region=dict(zip(['left', 'top', 'width', 'height'],
                        region)))


def splitArgs(args, split='_'):
    """Split a Namespace into a Namespace of Namespaces based on shared
    prefixes.  The string separating the prefix from the rest of the
    argument is determined by the optional "split" parameter.
    Parameters not containing the splitting string are kept as-is.

    """
    def splitKey(k):
        s = k.split(split, 1)
        return (None, s[0]) if len(s) == 1 else s

    Namespace = type(args)
    args = vars(args)
    firstKeys = {splitKey(k)[0] for k in args}
    result = Namespace()
    for k in firstKeys - {None}:
        setattr(result, k, Namespace())
    for k, v in args.items():
        f, s = splitKey(k)
        if f is None:
            setattr(result, s, v)
        else:
            setattr(getattr(result, f), s, v)
    return result


__all__ = (
    'get_stain_vector',
    'get_stain_matrix',
    'get_region_dict',
    'splitArgs',
)
