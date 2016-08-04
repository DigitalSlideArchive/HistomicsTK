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

    # transform input image to LAB color space
    imInputLAB = htk_color_conversion.RudermanLABFwd(imInput)

    # compute mean and stddev of input in LAB color space
    Mu = np.zeros(3)
    Sigma = np.zeros(3)

    for i in range(3):
        Mu[i] = imInputLAB[:, :, i].mean()
        Sigma[i] = (imInputLAB[:, :, i] - Mu[i]).std()

    # perform reinhard normalization
    imNmzd = htk_color_normalization.ReinhardNorm(imInput, Mu, Sigma)

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

    # get Hematoxylin and Eosin
    Hematoxylin = imDeconvolved.Stains[::2, ::2, 0]
    Eosin = imDeconvolved.Stains[::2, ::2, 1]

    #
    # Perform feature extraction
    #
    print('>> Performing feature extraction')

    df = htk_features.ExtractNuclearFeatures(
        imNucleiSegMask, Hematoxylin, Eosin, W, args.K, args.Fs, args.Delta)

    #
    # Create HDF5 file
    #
    print('>> Writing HDF5 file')

    hdf = pd.HDFStore(args.outputFile)
    hdf.put('d1', df, format='table', data_columns=True)

    print '--- Object x Features = ', hdf['d1'].shape


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
