import histomicstk.preprocessing.color_deconvolution as htk_cdeconv
import histomicstk.segmentation as htk_seg

import numpy as np

import skimage.color
import skimage.filters
import skimage.io
import skimage.measure
import skimage.morphology

from ctk_cli import CLIArgumentParser

import logging
logging.basicConfig()

stain_color_map = {
    'hematoxylin': [0.65, 0.70, 0.29],
    'eosin':       [0.07, 0.99, 0.11],
    'dab':         [0.27, 0.57, 0.78],
    'null':        [0.0, 0.0, 0.0]
}


def main(args):

    #
    # Read Input Image
    #
    print('>> Reading input image')

    im_input = skimage.io.imread(args.inputImageFile)[:, :, :3]

    #
    # Perform color deconvolution -unnormalized
    #
    print('>> Performing color deconvolution')

    stain_color_1 = stain_color_map[args.stain_1]
    stain_color_2 = stain_color_map[args.stain_2]
    stain_color_3 = stain_color_map[args.stain_3]

    w = np.array([stain_color_1, stain_color_2, stain_color_3]).T

    im_stains = htk_cdeconv.color_deconvolution(im_input, w).Stains

    im_membrane_stain = im_stains[:, :, 2].astype(np.float)

    #
    # Perform membrane detection
    #
    print('>> Performing membrane detection')

    Label, Split, Branches = htk_seg.membranes.membrane_detection(
        im_membrane_stain, min_sigma=args.filter_sigma_min,
        max_sigma=args.filter_sigma_max, beta=args.filter_beta,
        c=args.filter_c, f_threshold=args.foreground_threshold,
        min_lsize=args.min_labelsize, b_dilation=args.branch_merge_dilation,
        b_split_dilation=args.branch_split_dilation
    )

    # get x and y points of intersection between branches
    px, py = np.where(Branches > 0)

    #
    # Perform membrane color rendering
    #
    print('>> Performing membrane color rendering')

    # convert to rgb channel
    im_label = skimage.color.label2rgb(Label)
    im_label = 255 * im_label / im_label.max()

    split_points = np.where(Split > 0)

    red = im_input[:, :, 0]
    lred = im_label[:, :, 0]
    red[split_points] = lred[split_points]
    green = im_input[:, :, 1]
    lgreen = im_label[:, :, 1]
    green[split_points] = lgreen[split_points]
    blue = im_input[:, :, 2]
    lblue = im_label[:, :, 2]
    blue[split_points] = lblue[split_points]

    # generate membrane labeled image
    im_output = np.concatenate(
        (red[..., np.newaxis], green[..., np.newaxis], blue[..., np.newaxis]),
        axis=2
    )
    im_output[px, py, :] = 255

    #
    # Save output filtered image
    #
    print('>> Outputting membrane labeled image')

    skimage.io.imsave(args.outputMembraneLabelFile, im_output)


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
