from histomicstk.cli.utils import CLIArgumentParser
import numpy

from histomicstk.preprocessing.color_deconvolution import rgb_separate_stains_macenko_pca

from histomicstk.cli import utils


def main(args):
    args = utils.splitArgs(args)
    args.macenko.I_0 = numpy.array(args.macenko.I_0)

    utils.create_dask_client(args.dask)
    sample = utils.sample_pixels(args.sample)
    stain_matrix = rgb_separate_stains_macenko_pca(sample.T, **vars(args.macenko))
    with open(args.returnParameterFile, 'w') as f:
        for i, stain in enumerate(stain_matrix.T):
            f.write('stainColor_{} = {}\n'.format(i+1, ','.join(map(str, stain))))


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
