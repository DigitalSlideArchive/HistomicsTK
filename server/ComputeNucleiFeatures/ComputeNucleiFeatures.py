import os
import sys

import numpy as np
import pandas as pd
import scipy as sp
import skimage.io
import skimage.morphology

from ctk_cli import CLIArgumentParser

import histomicstk.preprocessing.color_conversion as htk_ccvt
import histomicstk.preprocessing.color_normalization as htk_cnorm
import histomicstk.preprocessing.color_deconvolution as htk_cdeconv
import histomicstk.filters.shape as htk_shape_filters
import histomicstk.segmentation as htk_seg
import histomicstk.features as htk_features

import logging
logging.basicConfig()

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), '..')))
from cli_common import utils  # noqa


def main(args):

    #
    # Read Input Image
    #
    print('>> Reading input image')

    im_input = skimage.io.imread(args.inputImageFile)[:, :, :3]

    #
    # Perform color normalization
    #
    print('>> Performing color normalization')

    # compute mean and stddev of input in LAB color space
    mu, sigma = htk_ccvt.lab_mean_std(im_input)

    # perform reinhard normalization
    im_nmzd = htk_cnorm.reinhard(im_input, mu, sigma)

    #
    # Perform color deconvolution
    #
    print('>> Performing color deconvolution')

    w = utils.get_stain_matrix(args)

    im_stains = htk_cdeconv.color_deconvolution(im_nmzd, w).Stains

    im_nuclei_stain = im_stains[:, :, 0].astype(np.float)

    #
    # Perform nuclei segmentation
    #
    print('>> Performing nuclei segmentation')

    # segment foreground
    im_fgnd_mask = sp.ndimage.morphology.binary_fill_holes(
        im_nuclei_stain < args.foreground_threshold)

    # run adaptive multi-scale LoG filter
    im_log = htk_shape_filters.clog(im_nuclei_stain, im_fgnd_mask,
                                    sigma_min=args.min_radius * np.sqrt(2),
                                    sigma_max=args.max_radius * np.sqrt(2))

    im_nuclei_seg_mask, seeds, max = htk_seg.nuclear.max_clustering(
        im_log, im_fgnd_mask, args.local_max_search_radius)

    # filter out small objects
    im_nuclei_seg_mask = htk_seg.label.area_open(
        im_nuclei_seg_mask, args.min_nucleus_area).astype(np.int)

    #
    # Perform feature extraction
    #
    print('>> Performing feature extraction')

    im_nuclei = im_stains[:, :, 0]

    if args.cytoplasm_features:
        im_cytoplasm = im_stains[:, :, 1]
    else:
        im_cytoplasm = None

    df = htk_features.ComputeNucleiFeatures(
        im_nuclei_seg_mask, im_nuclei, im_cytoplasm,
        fsd_bnd_pts=args.fsd_bnd_pts,
        fsd_freq_bins=args.fsd_freq_bins,
        cyto_width=args.cyto_width,
        num_glcm_levels=args.num_glcm_levels,
        morphometry_features_flag=args.morphometry_features,
        fsd_features_flag=args.fsd_features,
        intensity_features_flag=args.intensity_features,
        gradient_features_flag=args.gradient_features,
    )

    #
    # Create HDF5 file
    #
    print('>> Writing HDF5 file')

    hdf = pd.HDFStore(args.outputFile)
    hdf.put('d1', df, format='table', data_columns=True)

    print '--- Object x Features = ', hdf['d1'].shape


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
