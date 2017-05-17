import os
import sys

from ctk_cli import CLIArgumentParser
import numpy

from histomicstk.preprocessing.color_deconvolution import rgb_separate_stains_macenko_pca

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), '..')))
from cli_common import utils  # noqa


def main(args):
    args = utils.splitArgs(args)
    args.macenko.I_0 = numpy.array(args.macenko.I_0)

    utils.start_dask(args.dask)
    sample = utils.sample_pixels(args.sample)
    stain_matrix = rgb_separate_stains_macenko_pca(sample.T, **vars(args.macenko))
    with open(args.returnParameterFile, 'w') as f:
        for i, stain in enumerate(stain_matrix.T):
            f.write('stainColor_{} = {}\n'.format(i+1, ','.join(map(str, stain))))


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
