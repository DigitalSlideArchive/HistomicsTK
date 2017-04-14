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

    if args.maxRegionSize != -1:
        if useWholeImage:
            size = max(ts.sizeX, ts.sizeY)
        else:
            size = max(args.region[-2:])
        if size > args.maxRegionSize:
            raise ValueError('Requested region is too large!  Please see --maxRegionSize')

    im_input = ts.getRegion(format=large_image.tilesource.TILE_FORMAT_NUMPY,
                            **({} if useWholeImage else dict(
                                region=dict(zip(['left', 'top', 'width', 'height'],
                                                args.region)))))[0]

    # Create stain matrix
    print('>> Creating stain matrix')

    w = htk_cd.utils.get_stain_matrix(args)
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


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
