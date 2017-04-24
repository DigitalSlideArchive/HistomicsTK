from ctk_cli import CLIArgumentParser
from dask.distributed import Client
import numpy

from histomicstk.utils import sample_pixels
from histomicstk.preprocessing.color_deconvolution import rgb_separate_stains_macenko_pca


def main(args):
    returnParameterFile, args = splitArgs(args)
    args['macenko']['I_0'] = numpy.array(args['macenko']['I_0'])
    for k in 'magnification', 'sample_percent', 'sample_approximate_total':
        if args['sample'][k] == -1:
            del args['sample'][k]

    Client(args['dask']['scheduler_address'] or None)
    sample = sample_pixels(**args['sample'])
    stain_matrix = rgb_separate_stains_macenko_pca(sample.T, **args['macenko'])
    with open(returnParameterFile, 'w') as f:
        for i, stain in enumerate(stain_matrix.T):
            f.write('stainColor_{} = {}\n'.format(i+1, ','.join(map(str, stain))))


def splitArgs(args):
    """Split the arguments in the given Namespace by the part of their
    name before the first underscore.  Returns a dict of dicts, where
    the first key is the part of the name before the split and the
    second is the part after.  returnParameterFile is handled
    separately, and must be given.

    """
    def splitKey(k):
        return k.split('_', 1)

    rpf = args.returnParameterFile
    args = vars(args).copy()
    del args['returnParameterFile']
    firstKeys = {splitKey(k)[0] for k in args}
    a = {k: {} for k in firstKeys}
    for k, v in args.items():
        f, s = splitKey(k)
        a[f][s] = v
    return rpf, a


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
