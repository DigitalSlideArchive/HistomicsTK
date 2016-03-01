import numpy as np
import skimage.io
import histomicstk as htk

# Define Romanesco globals for the style checker
_tempdir = _tempdir              # noqa
inputImageFile = inputImageFile  # noqa
stainColor_1 = stainColor_1      # noqa
stainColor_2 = stainColor_2      # noqa
beta = beta                      # noqa

outputStainImageFile_1 = outputStainImageFile_1  # noqa
outputStainImageFile_2 = outputStainImageFile_2  # noqa
outputStainImageFile_3 = outputStainImageFile_3  # noqa

# Read Input Image
print('>> Reading input image')

print(inputImageFile)

inputImage = skimage.io.imread(inputImageFile)[:, :, :3]

# Create stain matrix
print('>> Creating stain matrix')

W_init = np.array([stainColor_1, stainColor_2]).T
print W_init

# Perform color deconvolution
print('>> Performing color deconvolution')
res = htk.SparseColorDeconvolution(inputImage, W_init, beta)
W_est = np.concatenate((res.W, np.zeros((3, 1))), 1)
res = htk.ColorDeconvolution(inputImage, W_est)

# write stain images to output
print('>> Outputting individual stain images')

print outputStainImageFile_1
skimage.io.imsave(outputStainImageFile_1, res.Stains[:, :, 0])

print outputStainImageFile_2
skimage.io.imsave(outputStainImageFile_2, res.Stains[:, :, 1])

print outputStainImageFile_3
skimage.io.imsave(outputStainImageFile_3, res.Stains[:, :, 2])
