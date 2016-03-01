import os
import numpy as np
import skimage.io
import histomicstk as htk

# Define Girder Worker globals for the style checker
inputImageFile = inputImageFile  # noqa
_tempdir = _tempdir   # noqa

# Read Input Image
print('>> Reading input image')
print(inputImageFile)
inputImage = skimage.io.imread(inputImageFile)

# Perform color deconvolution
print('>> Performing color deconvolution')
W = np.array([[0.650, 0.072, 0], [0.704, 0.990, 0], [0.286, 0.105, 0]])
res = htk.ColorDeconvolution(inputImage, W)

# write stain images to output
print('>> Outputting individual stain images')
outFileSuffix = os.path.split(inputImageFile)[1]

outputStainImageFile_1 = os.path.join(_tempdir, 'stain_1_' + outFileSuffix)
skimage.io.imsave(outputStainImageFile_1, res.Stains[:, :, 0])

outputStainImageFile_2 = os.path.join(_tempdir, 'stain_2_' + outFileSuffix)
skimage.io.imsave(outputStainImageFile_2, res.Stains[:, :, 1])

outputStainImageFile_3 = os.path.join(_tempdir, 'stain_3_' + outFileSuffix)
skimage.io.imsave(outputStainImageFile_3, res.Stains[:, :, 2])
