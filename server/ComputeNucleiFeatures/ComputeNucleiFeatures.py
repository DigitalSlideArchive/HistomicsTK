import numpy as np
import pandas as pd
import scipy as sp
import skimage.io
import skimage.morphology

from ctk_cli import CLIArgumentParser

import histomicstk.preprocessing.color_conversion as htk_color_conversion
import histomicstk.preprocessing.color_normalization as htk_color_normalization
import histomicstk.preprocessing.color_deconvolution as htk_color_deconvolution
import histomicstk.filters.shape as htk_shape_filters
import histomicstk.segmentation as htk_seg
import histomicstk.features as htk_features

import logging
logging.basicConfig()

stainColorMap = {
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

    imInput = skimage.io.imread(args.inputImageFile)[:, :, :3]

    #
    # Perform color normalization
    #
    print('>> Performing color normalization')

    # compute mean and stddev of input in LAB color space
    Mu, Sigma = htk_color_conversion.compute_lab_mean_std(imInput)

    # perform reinhard normalization
    imNmzd = htk_color_normalization.reinhard(imInput, Mu, Sigma)

    #
    # Perform color deconvolution
    #
    print('>> Performing color deconvolution')

    stainColor_1 = stainColorMap[args.stain_1]
    stainColor_2 = stainColorMap[args.stain_2]
    stainColor_3 = stainColorMap[args.stain_3]

    W = np.array([stainColor_1, stainColor_2, stainColor_3]).T

    imDeconvolved = htk_color_deconvolution.ColorDeconvolution(imNmzd, W)

    imNucleiStain = imDeconvolved.Stains[::2, ::2, 0].astype(np.float)

    #
    # Perform nuclei segmentation
    #
    print('>> Performing nuclei segmentation')

    # segment foreground
    imFgndMask = sp.ndimage.morphology.binary_fill_holes(
        imNucleiStain < args.foreground_threshold)

    # run adaptive multi-scale LoG filter
    imLog = htk_shape_filters.cLoG(imNucleiStain, imFgndMask,
                                   SigmaMin=args.min_radius * np.sqrt(2),
                                   SigmaMax=args.max_radius * np.sqrt(2))

    imNucleiSegMask, Seeds, Max = htk_seg.nuclear.MaxClustering(
        imLog, imFgndMask, args.local_max_search_radius)

    # filter out small objects
    imNucleiSegMask = htk_seg.label.FilterLabel(
        imNucleiSegMask, Lower=args.min_nucleus_area).astype(np.int)

    #
    # Perform feature extraction
    #
    print('>> Performing feature extraction')

    im_nuclei = imDeconvolved.Stains[::2, ::2, 0]

    if args.cytoplasm_features:
        im_cytoplasm = imDeconvolved.Stains[::2, ::2, 1]
    else:
        im_cytoplasm = None

    df = htk_features.ComputeNucleiFeatures(
        imNucleiSegMask, im_nuclei, im_cytoplasm,
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
