import os
import sys

import numpy as np
import skimage.io

from ctk_cli import CLIArgumentParser

import histomicstk.preprocessing.color_deconvolution as htk_cdeconv

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), '..')))
from cli_common import utils  # noqa


def main(args):

    # Read Input Image
    print('>> Reading input image')

    print(args.inputImageFile)

    im_input = skimage.io.imread(args.inputImageFile)[:, :, :3]

    # Create stain matrix
    print('>> Creating stain matrix')

    w_init = utils.get_stain_matrix(args, 2)

    print w_init

    # Perform color deconvolution
    print('>> Performing color deconvolution')

    res = htk_cdeconv.sparse_color_deconvolution(
        im_input, w_init, args.beta)
    w_est = np.concatenate((res.Wc, np.zeros((3, 1))), 1)
    res = htk_cdeconv.color_deconvolution(im_input, w_est)

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
