import skimage.io

from histomicstk.cli.utils import CLIArgumentParser
import large_image

import histomicstk.preprocessing.color_deconvolution as htk_cd

from histomicstk.cli import utils

import logging
logging.basicConfig()


def main(args):

    # Read Input Image
    print('>> Reading input image')

    print(args.inputImageFile)

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


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
