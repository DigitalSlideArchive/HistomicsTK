from ctk_cli import CLIArgumentParser
import numpy

from histomicstk.utils import sample_pixels
from histomicstk.preprocessing.color_deconvolution import rgb_macenko_stain_matrix

def main(args):
    returnParameterFile, args = splitArgs(args)
    args['macenko']['I_0'] = numpy.array(args['macenko']['I_0'])

    sample = sample_pixels(**args['sample'])
    stain_matrix = rgb_macenko_stain_matrix(sample.T, **args['macenko'])
    with open(returnParameterFile, 'w') as f:
        for i, stain in enumerate(stain_matrix.T):
            f.write('stainColor_{} = {}\n'.format(i+1, ','.join(map(str, stain))))

def splitArgs(args):
    """Split the arguments in the given Namespace by the part of their
    name before the first underscore.  Returns a dict of dicts, where
    the first key is the part of the name before the split and the
    second is the part after.  returnParameterFile is handled
    separately, and must be given.  Other keys with value None are
    removed to permit default parameters to work.

    """
    rpf = args.returnParameterFile
    args = vars(args).copy()
    del args['returnParameterFile']
    def splitKey(k): return k.split('_', 1)
    firstKeys = {splitKey(k)[0] for k in args}
    a = {k: {} for k in firstKeys}
    for k, v in args.items():
        if v is None:
            continue
        f, s = splitKey(k)
        a[f][s] = v
    return rpf, a

if __name__=='__main__':
    main(CLIArgumentParser().parse_args())
