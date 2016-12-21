import numpy as np
import skimage.io

from ctk_cli import CLIArgumentParser

import histomicstk.preprocessing.color_deconvolution as htk_cd

import logging
logging.basicConfig()


def main(args):

    # Read Input Image
    print('>> Reading input image')

    print(args.inputImageFile)

    im_input = skimage.io.imread(args.inputImageFile)

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

if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
