import os
import sys

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

    res = htk_cdeconv.separate_stains_xu_snmf(im_input, w_init, args.beta)
    w_est = htk_cdeconv.complement_stain_matrix(res.Wc)

    with open(args.returnParameterFile, 'w') as f:
        for i, stain in enumerate(w_est):
            f.write('stainColor_{} = {}\n'.format(i+1, ','.join(map(str, stain))))


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
