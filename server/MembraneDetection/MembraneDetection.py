import histomicstk.preprocessing.color_deconvolution as htk_cdeconv
import histomicstk.filters.shape as htk_shape_filters

import numpy as np
import scipy as sp

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
    size_min = min(im_input.shape[0], im_input.shape[1])
    im_input = im_input[0:size_min, 0:size_min, :]

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
    # Perform membrane enhancement
    #
    print('>> Performing membrane enhancement')

    # membrane filtering
    im_deviation, im_thetas = htk_shape_filters.membranefilter(
        im_membrane_stain, [1, 2, 3], 4, 2
    )
    im_deviation = 255 * im_deviation / im_deviation.max()

    # segmentation
    im_mask = im_deviation > args.foreground_threshold* \
        skimage.filters.threshold_otsu(im_deviation)
    im_opened = sp.ndimage.binary_opening(
        im_mask, structure=np.ones((3,3))
    ).astype(np.int)

    # skeletonization
    im_skeleton = skimage.morphology.skeletonize(im_opened)

    #
    # Perform membrane label detection
    #
    print('>> Performing membrane label detection')

    # set default branch mask
    branch_mask = np.ones((3,3))

    # perform convolution
    im_branch_mask = sp.signal.convolve2d(
        im_skeleton, branch_mask, boundary='wrap', mode='same'
    )

    im_branches = np.zeros_like(im_branch_mask)
    im_branches[np.where(im_branch_mask > args.branch_threshold)] = 1

    # label detection
    im_split = im_skeleton & ~sp.ndimage.binary_dilation(
        im_branches,
        structure=skimage.morphology.disk(args.branch_split_dilation)
    )
    im_split = sp.ndimage.binary_dilation(
        im_split,
        structure=skimage.morphology.disk(args.branch_merge_dilation)
    )

    # get labeled arrays
    labeled_array, num_features = sp.ndimage.label(im_split)

    # get labels
    im_label = skimage.color.label2rgb(labeled_array)
    im_label = 255 * im_label / im_label.max();

    # get x and y points of intersection between branches
    px, py = np.where(im_branches > 0)

    #
    # Perform membrane color rendering
    #
    print('>> Performing membrane color rendering')

    split_points = np.where(im_split>0)

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
    im_membraned_color = np.concatenate(
        (red[...,np.newaxis], green[...,np.newaxis], blue[...,np.newaxis]),
        axis=2
    )
    im_membraned_color[px, py, :] = 255

    #
    # Save output filtered image
    #
    print('>> Outputting membrane labeled image')

    skimage.io.imsave(args.outputMembraneLabelFile, im_membraned_color)


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
