# Tests SparseColorDeconvolution.py. Uses a standard H&E calibration matrix as
# the initialization for adaptive, sparse color deconvolution. The stain
# intensity images obtained from sparse (adaptive) and normal color
# deconvolution are displayed side-by-side for comparison.
# Also tests helper functions:
# ComplementStainMatrix.py
# OpticalDensityFwd.py
# OpticalDensityInv.py

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy
import skimage.io as io

# Define globals for the style checker
# TODO: Ensure these are actually provided
ColorDeconvolution = ColorDeconvolution  # noqa
SparseColorDeconvolution = SparseColorDeconvolution  # noqa

# Define input image, initial H&E color matrix, and sparsity coefficient
File = './A.png'
W = numpy.array([[0.650, 0.072, 0],
                 [0.704, 0.990, 0],
                 [0.286, 0.105, 0]])
Beta = 5e-2

# open image
I = io.imread(File)

# remove alpha channel
I = I[:, :, :3]

# perform standard color deconvolution
Static = ColorDeconvolution(I, W)

# perform adaptive color deconvolution
Adaptive = SparseColorDeconvolution(I, W[:, :2], Beta)
Wa = numpy.concatenate((Adaptive.W, numpy.zeros((3, 1))), 1)
Adaptive = ColorDeconvolution(I, Wa)

# view output - check if coherent
plt.figure
plt.subplot(3, 2, 1)
plt.imshow(I)
plt.title('Color Image')
plt.subplot(3, 2, 3)
plt.imshow(Static.Stains[:, :, 0], cmap=cm.Greys_r)
plt.title('Hematoxylin - static')
plt.subplot(3, 2, 4)
plt.imshow(Adaptive.Stains[:, :, 0], cmap=cm.Greys_r)
plt.title('Hematoxylin - adaptive')
plt.subplot(3, 2, 5)
plt.imshow(Static.Stains[:, :, 1], cmap=cm.Greys_r)
plt.title('Eosin - static')
plt.subplot(3, 2, 6)
plt.imshow(Adaptive.Stains[:, :, 1], cmap=cm.Greys_r)
plt.title('Eosin - adaptive')
