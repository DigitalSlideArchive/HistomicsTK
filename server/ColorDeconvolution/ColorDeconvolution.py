import os

import numpy as np
import skimage.io

from ctk_cli import CLIArgumentParser
import large_image

import histomicstk.preprocessing.color_deconvolution as htk_cd

import logging
logging.basicConfig()


def main(args):

    # Read Input Image
    print('>> Reading input image')

    print(args.inputImageFile)

    ts = large_image.getTileSource(args.inputImageFile)
    if len(args.region) != 4:
        raise ValueError('Exactly four values required for --region')
    useWholeImage = args.region == [-1] * 4
    im_input = ts.getRegion(format=large_image.tilesource.TILE_FORMAT_NUMPY,
                            **({} if useWholeImage else dict(
                                region=dict(zip(['left', 'top', 'width', 'height'],
                                                args.region)))))[0]

    # Create stain matrix
    print('>> Creating stain matrix')

    w = np.array([args.stainColor_1, args.stainColor_2, args.stainColor_3]).T
    print w

    # Perform color deconvolution
    print('>> Performing color deconvolution')
    im_stains = htk_cd.color_deconvolution(im_input, w).Stains

    # write stain images to output
    print('>> Outputting individual stain images')

    print args.outputStainImageFile_1
    skimage.io.imsave(args.outputStainImageFile_1, im_stains[:, :, 0])

    print args.outputStainImageFile_2
    skimage.io.imsave(args.outputStainImageFile_2, im_stains[:, :, 1])

    print args.outputStainImageFile_3
    skimage.io.imsave(args.outputStainImageFile_3, im_stains[:, :, 2])

    if args.returnParameterFile is not None:
        with open(args.returnParameterFile, 'w') as f:
            f.write('region = ' + ','.join(map(str, args.region)) + '\n')
            for i in range(1, 4):
                name = 'outputStainImageFile_{}'.format(i)
                # Just use the absolute path
                f.write('{} = {}\n'.format(name, os.path.abspath(getattr(args, name))))

if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
