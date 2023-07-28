import json
import logging
import numpy as np
from pathlib import Path

import large_image

import histomicstk
import histomicstk.preprocessing.color_deconvolution as htk_cd
from histomicstk.cli import utils
from histomicstk.cli.utils import CLIArgumentParser

logging.basicConfig()

def colordeconvolve(args):

    import skimage.io
    ts = large_image.getTileSource(args.inputImageFile)

    im_input = ts.getRegion(
        format=large_image.tilesource.TILE_FORMAT_NUMPY,
        **utils.get_region_dict(args.region, args.maxRegionSize, ts)
    )[0]

    # Create stain matrix
    print('>> Creating stain matrix')

    w = utils.get_stain_matrix(args)
    print(w)

    # Perform color deconvolution
    print('>> Performing color deconvolution')
    im_stains = htk_cd.color_deconvolution(im_input, w).Stains

    # write stain images to output
    print('>> Outputting individual stain images')

    print(args.outputStainImageFile_1)
    skimage.io.imsave(args.outputStainImageFile_1, im_stains[:, :, 0])

    print(args.outputStainImageFile_2)
    skimage.io.imsave(args.outputStainImageFile_2, im_stains[:, :, 1])

    print(args.outputStainImageFile_3)
    skimage.io.imsave(args.outputStainImageFile_3, im_stains[:, :, 2])

    region = utils.get_region_dict(args.region, args.maxRegionSize, ts)['region']

    return region

def colordeconvolve_WSI(args):
    sink0 = large_image.new()
    sink1 = large_image.new()
    sink2 = large_image.new()
    ts = large_image.getTileSource(args.inputImageFile)

    # Create stain matrix
    print('>> Creating stain matrix')

    w = utils.get_stain_matrix(args)
    print(w)

    for tile in ts.tileIterator():
        im_stains = htk_cd.color_deconvolution(tile['tile'], w).Stains
        sink0.addTile(im_stains[:, :, 0], tile['x'], tile['y'])
        sink1.addTile(im_stains[:, :, 1], tile['x'], tile['y'])
        sink2.addTile(im_stains[:, :, 2], tile['x'], tile['y'])
    sink0.write(args.outputStainImageFile_1)
    sink1.write(args.outputStainImageFile_2)
    sink2.write(args.outputStainImageFile_3)


    region = utils.get_region_dict(args.region, args.maxRegionSize, ts)
    print(region)

    return region

def main(args):


    # Read Input Image
    print('>> Reading input image')

    print(args.inputImageFile)

    if np.all(np.array(args.region) == -1):
        region = colordeconvolve_WSI(args)
    else:
        region = colordeconvolve_ROI(args)

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
            }
        }]
        if args.stain_3 == 'null':
            annotation[2:] = []
        if args.stain_2 == 'null':
            annotation[1:2] = []

        with open(args.outputAnnotationFile, 'w') as annotation_file:
            json.dump(annotation, annotation_file, separators=(',', ':'), sort_keys=False)


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
