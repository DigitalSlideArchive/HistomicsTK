import os
import numpy as np
import skimage.io
import histomicstk as htk

# Define Romanesco globals for the style checker
inputImageFile = inputImageFile  # noqa
stainColor_1 = stainColor_1      # noqa
stainColor_2 = stainColor_2      # noqa
beta = beta                      # noqa
_tempdir = _tempdir              # noqa

# Read Input Image
print('>> Reading input image')

print(inputImageFile)

inputImage = skimage.io.imread(inputImageFile)

# Create stain matrix
print('>> Creating stain matrix')

W_init = np.array([stainColor_1, stainColor_2]).T
print W_init

# Perform color deconvolution
print('>> Performing color deconvolution')
res = htk.SparseColorDeconvolution(inputImage, W_init, beta)
Wa = np.concatenate((res.W, np.zeros((3, 1))), 1)
res = htk.ColorDeconvolution(inputImage, Wa)

# write stain images to output
print('>> Outputting individual stain images')
outFileSuffix = os.path.split(inputImageFile)[1]

outputStainImageFile_1 = os.path.join(_tempdir, 'stain_1_' + outFileSuffix)
skimage.io.imsave(outputStainImageFile_1, res.Stains[:, :, 0])

outputStainImageFile_2 = os.path.join(_tempdir, 'stain_2_' + outFileSuffix)
skimage.io.imsave(outputStainImageFile_2, res.Stains[:, :, 1])

outputStainImageFile_3 = os.path.join(_tempdir, 'stain_3_' + outFileSuffix)
skimage.io.imsave(outputStainImageFile_3, res.Stains[:, :, 2])
