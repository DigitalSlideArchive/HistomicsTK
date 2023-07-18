import json
import numpy

from pathlib import Path

import histomicstk
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

    # record stain color metadata
    stain_color_metadata = []
    for i, stain in enumerate(w_est.T):
        stain_color_metadata = 'stainColor_{} = {}\n'.format(i + 1, ','.join(map(str, stain)))

    annotation = {
        'name': 'SeperateStainsXuSnmf',
        'stainColor': stain_color_metadata,
        'attributes': {
            'params': vars(args),
            'cli': Path(__file__).stem,
            'version': histomicstk.__version__,
        },
    }

    with open(args.returnParameterFile, 'w') as metadata_file:
        json.dump(annotation, metadata_file, seperators=(',', ':'), sort_keys=False)


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
