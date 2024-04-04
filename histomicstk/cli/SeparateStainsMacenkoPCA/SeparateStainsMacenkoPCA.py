import json
from pathlib import Path

import numpy

import histomicstk
from histomicstk.cli import utils
from histomicstk.cli.utils import CLIArgumentParser
from histomicstk.preprocessing.color_deconvolution import \
    rgb_separate_stains_macenko_pca


def main(origargs):
    args = utils.splitArgs(origargs)
    args.macenko.I_0 = numpy.array(args.macenko.I_0)

    utils.create_dask_client(args.dask)
    sample = utils.sample_pixels(args.sample)
    stain_matrix = rgb_separate_stains_macenko_pca(sample.T, **vars(args.macenko))

    # record metadata
    annotation = {
        'name': 'SeperateStainsMacenkoPCA',
        'attributes': {
            'params': vars(origargs),
            'cli': Path(__file__).stem,
            'version': histomicstk.__version__,
        },
    }
    for i, stain in enumerate(stain_matrix.T):
        annotation['attributes'][f'stainColor_{i + 1}'] = stain.tolist()

    with open(args.outputAnnotationFile, 'w') as annotation_file:
        json.dump(annotation, annotation_file, sort_keys=False)


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
