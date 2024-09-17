import json
import logging
from pathlib import Path

import large_image
import numpy as np

import histomicstk
import histomicstk.preprocessing.color_deconvolution as htk_cd
from histomicstk.cli import utils
from histomicstk.cli.utils import CLIArgumentParser

logging.basicConfig()


def colorDeconvolve(args):
    sink0 = large_image.new()
    sink1 = large_image.new()
    sink2 = large_image.new()
    ts = large_image.getTileSource(args.inputImageFile)
    region = {}
    it_kwargs = {}
    tileSize = 8192
    it_kwargs['tile_size'] = dict(width=tileSize, height=tileSize)

    # Provides crop area if ROI present in arguments
    if np.all(np.array(args.region) != -1):
        it_kwargs['region'] = {
            'left': args.region[0],
            'top': args.region[1],
            'width': args.region[2],
            'height': args.region[3],
            'units': 'base_pixels',
        }
        sink0.crop = tuple(args.region)
        sink1.crop = tuple(args.region)
        sink2.crop = tuple(args.region)
        region = utils.get_region_dict(args.region, None, ts)['region']

    # Create stain matrix
    print('>> Creating stain matrix')

    w = utils.get_stain_matrix(args)
    print(w)

    # Perform color deconvolution
    print('>> Performing color deconvolution')
    for tile in ts.tileIterator(**it_kwargs):
        im_stains = htk_cd.color_deconvolution(tile['tile'], w).Stains
        sink0.addTile(im_stains[:, :, 0], tile['x'], tile['y'])
        sink1.addTile(im_stains[:, :, 1], tile['x'], tile['y'])
        sink2.addTile(im_stains[:, :, 2], tile['x'], tile['y'])

    # Write stain images to output
    print('>> Outputting individual stain images')

    if args.outputStainImageFile_1:
        print(args.outputStainImageFile_1)
        sink0.write(args.outputStainImageFile_1)
    if args.outputStainImageFile_2:
        print(args.outputStainImageFile_2)
        sink1.write(args.outputStainImageFile_2)
    if args.outputStainImageFile_3:
        print(args.outputStainImageFile_3)
        sink2.write(args.outputStainImageFile_3)

    return region


def main(args):

    # Read Input Image
    print('>> Reading input image')

    print(args.inputImageFile)

    region = colorDeconvolve(args)

    if args.outputAnnotationFile:

        annotation = [{
            'name': 'Deconvolution %s' % (
                args.stain_1 if args.stain_1 != 'custom' else str(args.stain_1_vector)),
            'elements': [{
                'type': 'image',
                'girderId': 'outputStainImageFile_1',
                'transform': {
                    'xoffset': region.get('left', 0),
                    'yoffset': region.get('top', 0),
                },
            }],
            'attributes': {
                'cli': Path(__file__).stem,
                'params': vars(args),
                'version': histomicstk.__version__,
            },
        }, {
            'name': 'Deconvolution %s' % (
                args.stain_2 if args.stain_2 != 'custom' else str(args.stain_2_vector)),
            'elements': [{
                'type': 'image',
                'girderId': 'outputStainImageFile_2',
                'transform': {
                    'xoffset': region.get('left', 0),
                    'yoffset': region.get('top', 0),
                },
            }],
            'attributes': {
                'cli': Path(__file__).stem,
                'params': vars(args),
                'version': histomicstk.__version__,
            },
        }, {
            'name': 'Deconvolution %s' % (
                args.stain_3 if args.stain_3 != 'custom' else str(args.stain_3_vector)),
            'elements': [{
                'type': 'image',
                'girderId': 'outputStainImageFile_3',
                'transform': {
                    'xoffset': region.get('left', 0),
                    'yoffset': region.get('top', 0),
                },
            }],
            'attributes': {
                'cli': Path(__file__).stem,
                'params': vars(args),
                'version': histomicstk.__version__,
            },
        }]
        if args.stain_3 == 'null':
            annotation[2:] = []
        if args.stain_2 == 'null':
            annotation[1:2] = []

        if args.outputAnnotationFile:
            with open(args.outputAnnotationFile, 'w') as annotation_file:
                json.dump(annotation, annotation_file, separators=(',', ':'), sort_keys=False)


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
