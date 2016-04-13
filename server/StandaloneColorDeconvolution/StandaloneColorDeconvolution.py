#!/usr/bin/env python
from ctk_cli import CLIArgumentParser
import logging
logging.basicConfig()

import numpy as np
import skimage.io
import histomicstk as htk


def main(args):

    # Read Input Image
    print('>> Reading input image')

    print(args.inputImageFile)

    inputImage = skimage.io.imread(args.inputImageFile)

    # Create stain matrix
    print('>> Creating stain matrix')

    W = np.array([args.stainColor_1, args.stainColor_2, args.stainColor_3]).T
    print W

    # Perform color deconvolution
    print('>> Performing color deconvolution')
    res = htk.ColorDeconvolution(inputImage, W)

    # write stain images to output
    print('>> Outputting individual stain images')

    print args.outputStainImageFile_1
    skimage.io.imsave(args.outputStainImageFile_1, res.Stains[:, :, 0])

    print args.outputStainImageFile_2
    skimage.io.imsave(args.outputStainImageFile_2, res.Stains[:, :, 1])

    print args.outputStainImageFile_3
    skimage.io.imsave(args.outputStainImageFile_3, res.Stains[:, :, 2])


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())

