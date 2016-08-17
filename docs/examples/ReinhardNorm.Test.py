# Tests ReinhardNorm.py. Normalizes an input image to a standard image in LAB
# color space. The input image, standard image, and normalized output image are
# displayed side-by-side.
# Also tests helper functions:
# rgb_to_lab.py
# rgb_lab.py

import matplotlib.pyplot as plt
import skimage.io as io

# Define globals for the style checker
# TODO: Ensure these are actually provided
ReinhardNorm = ReinhardNorm  # noqa
RudermanLABFwd = RudermanLABFwd  # noqa

# Define input image, initial H&E color matrix, and sparsity coefficient
InputFile = './testing/A.png'
StandardFile = './testing/B.png'

# open images
Input = io.imread(InputFile)
Standard = io.imread(StandardFile)

# strip out alpha channels
Input = Input[:, :, :3]
Standard = Standard[:, :, :3]

# calculate mean, SD of standard image in LAB color space
LABStandard = RudermanLABFwd(Standard)
m = Standard.shape[0]
n = Standard.shape[1]
Mu = LABStandard.sum(axis=0).sum(axis=0) / (m * n)
LABStandard[:, :, 0] = LABStandard[:, :, 0] - Mu[0]
LABStandard[:, :, 1] = LABStandard[:, :, 1] - Mu[1]
LABStandard[:, :, 2] = LABStandard[:, :, 2] - Mu[2]
Sigma = ((LABStandard * LABStandard).sum(axis=0).sum(axis=0) /
         (m * n - 1)) ** 0.5
LABStandard[:, :, 0] = LABStandard[:, :, 0] / Sigma[0]
LABStandard[:, :, 1] = LABStandard[:, :, 1] / Sigma[1]
LABStandard[:, :, 2] = LABStandard[:, :, 2] / Sigma[2]

# normalize input image
Normalized = ReinhardNorm(Input, Mu, Sigma)

# view output - check if coherent
plt.figure
plt.subplot(3, 1, 1)
plt.imshow(Standard)
plt.title('Standard Color Image')
plt.subplot(3, 1, 2)
plt.imshow(Input)
plt.title('Input Image')
plt.subplot(3, 1, 3)
plt.imshow(Normalized)
plt.title('Normalized')
