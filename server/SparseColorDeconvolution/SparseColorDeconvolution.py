import numpy as np
import skimage.io
import histomicstk as htk
from ctk_cli import CLIArgumentParser


def main(args):

    # Read Input Image
    print('>> Reading input image')

    print(args.inputImageFile)

    inputImage = skimage.io.imread(args.inputImageFile)[:, :, :3]

    # Create stain matrix
    print('>> Creating stain matrix')

    W_init = np.array([args.stainColor_1, args.stainColor_2]).T
    print W_init

    # Perform color deconvolution
    print('>> Performing color deconvolution')
    res = htk.SparseColorDeconvolution(inputImage, W_init, args.beta)
    W_est = np.concatenate((res.W, np.zeros((3, 1))), 1)
    res = htk.ColorDeconvolution(inputImage, W_est)

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
