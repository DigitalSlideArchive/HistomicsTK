import json
from pathlib import Path

import numpy

import histomicstk
from histomicstk.cli import utils
from histomicstk.cli.utils import CLIArgumentParser
from histomicstk.preprocessing.color_deconvolution import \
    rgb_separate_stains_macenko_pca


def main(args):
    args = utils.splitArgs(args)
    args.macenko.I_0 = numpy.array(args.macenko.I_0)

    utils.create_dask_client(args.dask)
    sample = utils.sample_pixels(args.sample)
    stain_matrix = rgb_separate_stains_macenko_pca(sample.T, **vars(args.macenko))

    # record stain color metadata
    stain_color_metadata = []
    for i, stain in enumerate(stain_matrix.T):
        stain_color_metadata = 'stainColor_{} : {}\n'.format(i + 1, ','.join(map(str, stain)))

    annotation = {
        'name': 'SeperateStainsMacenkoPCA',
        'stain_color': stain_color_metadata,
        'attributes': {
            'params': vars(args),
            'return_parameters': args.returnParameterFile,
            'cli': Path(__file__).stem,
            'version': histomicstk.__version__,
        },
    }

    with open(args.returnParameterFile, 'w') as metadata_file:
        json.dump(annotation, metadata_file, seperators=(',', ':'), sort_keys=False)


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
