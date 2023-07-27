import json
from pathlib import Path

import numpy

import histomicstk
import histomicstk.preprocessing.color_deconvolution as htk_cdeconv
from histomicstk.cli import utils
from histomicstk.cli.utils import CLIArgumentParser


def main(origargs):
    args = utils.splitArgs(origargs)
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

    annotation = {
        'name': 'SeperateStainsXuSnmf',
        'attributes': {
            'params': vars(origargs),
            'cli': Path(__file__).stem,
            'version': histomicstk.__version__,
        },
    }
    for i, stain in enumerate(w_est.T):
        annotation['attributes']['stainColor_{}'.format(i + 1)] = stain.tolist()
    with open(args.outputAnnotationFile, 'w') as annotation_file:
        json.dump(annotation, annotation_file, sort_keys=False)


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
