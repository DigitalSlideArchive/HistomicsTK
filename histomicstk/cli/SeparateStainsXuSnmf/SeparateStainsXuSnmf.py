import numpy

import histomicstk.preprocessing.color_deconvolution as htk_cdeconv
from histomicstk.cli import utils
from histomicstk.cli.utils import CLIArgumentParser


def main(args):
    args = utils.splitArgs(args)
    args.snmf.I_0 = numpy.array(args.snmf.I_0)

    print('>> Starting Dask cluster and sampling pixels')
    utils.create_dask_client(args.dask)
    sample = utils.sample_pixels(args.sample)

    # Create stain matrix
    print('>> Creating stain matrix')

    args.snmf.w_init = utils.get_stain_matrix(args.stains, 2)

    print(args.snmf.w_init)

    # Perform color deconvolution
    print('>> Performing color deconvolution')

    w_est = htk_cdeconv.rgb_separate_stains_xu_snmf(sample.T, **vars(args.snmf))
    w_est = htk_cdeconv.complement_stain_matrix(w_est)

    with open(args.returnParameterFile, 'w') as f:
        for i, stain in enumerate(w_est.T):
            f.write('stainColor_{} = {}\n'.format(i + 1, ','.join(map(str, stain))))


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
