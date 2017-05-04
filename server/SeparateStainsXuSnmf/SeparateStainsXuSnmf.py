import os
import sys

from dask.distributed import Client
import numpy

from ctk_cli import CLIArgumentParser

import histomicstk.preprocessing.color_deconvolution as htk_cdeconv
from histomicstk.utils import sample_pixels

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), '..')))
from cli_common import utils  # noqa


def main(args):
    args = utils.splitArgs(args)
    args.snmf.I_0 = numpy.array(args.snmf.I_0)
    for k in 'magnification', 'sample_fraction', 'sample_approximate_total':
        if getattr(args.sample, k) == -1:
            delattr(args.sample, k)

    print(">> Starting Dask cluster and sampling pixels")
    Client(args.dask.scheduler_address or None)
    sample = sample_pixels(**vars(args.sample))

    # Create stain matrix
    print('>> Creating stain matrix')

    args.snmf.w_init = utils.get_stain_matrix(args.stains, 2)

    print args.snmf.w_init

    # Perform color deconvolution
    print('>> Performing color deconvolution')

    w_est = htk_cdeconv.rgb_separate_stains_xu_snmf(sample.T, **vars(args.snmf))
    w_est = htk_cdeconv.complement_stain_matrix(w_est)

    with open(args.returnParameterFile, 'w') as f:
        for i, stain in enumerate(w_est.T):
            f.write('stainColor_{} = {}\n'.format(i+1, ','.join(map(str, stain))))


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
