import numpy as np
import pandas
import scipy
import skimage.io
import skimage.morphology

import histomicstk as htk
from ctk_cli import CLIArgumentParser

import logging
logging.basicConfig()


def main(args):

    # Read Input Image
    print('>> Reading input image')

    print '--- Image Name = ', args.inputImageFile

    Standard = skimage.io.imread(args.inputImageFile)[:, :, :3]
    print '--- Original Image Size = ', Standard.shape

    # Reduce image size for testing
    Standard = Standard[50:200, 50:200]
    print '--- T by T Image Size = ', Standard.shape

    # Create stain matrix
    print('>> Creating stain matrix')
    W = np.array([args.stainColor_1, args.stainColor_2, args.stainColor_3]).T

    # Perform color deconvolution
    print('>> Performing color deconvolution')
    UNS = htk.ColorDeconvolution(Standard, W)

    # Constrained log filtering=== - generate R_{N}(x,y)
    Nuclei = UNS.Stains[::2, ::2, 0].astype(dtype=np.float)
    Mask = scipy.ndimage.morphology.binary_fill_holes(Nuclei < 160)

    Response = htk.cLoG(Nuclei, Mask, SigmaMin=4*1.414, SigmaMax=7*1.414)
    Label, Seeds, Max = htk.MaxClustering(Response.copy(), Mask, 10)
    Filtered = htk.FilterLabel(Label, 4, 80, None).astype(dtype=np.int)

    # Get Hematoxylin and Eosin
    Hematoxylin = UNS.Stains[::2, ::2, 0]
    Eosin = UNS.Stains[::2, ::2, 1]

    # Perform feature extraction
    print('>> Performing feature extraction')
    df = htk.FeatureExtraction(Filtered, Hematoxylin, Eosin, W)

    print('>> Writing HDF5 file')
    hdf = pandas.HDFStore(args.outputFile)
    hdf.put('d1', df, format='table', data_columns=True)
    print '--- Object x Features = ', hdf['d1'].shape


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
