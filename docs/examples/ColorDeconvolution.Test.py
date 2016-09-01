# Tests ColorDeconvolution.py. Uses a standard H&E calibration matrix to
# deconvolve a test image into hematoxylin and eosin stains, then displays the
# intensity images of these stains with the original.
# Also tests helper functions:
# ComplementStainMatrix.py
# rgb_to_od.py
# OpticalDensityInv.py

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy
import skimage.io as io

# Define globals for the style checker
# TODO: Ensure these are actually provided
ColorDeconvolution = ColorDeconvolution  # noqa

# Define input image and H&E color matrix
File = './B.png'
W = numpy.array([[0.650, 0.072, 0],
                 [0.704, 0.990, 0],
                 [0.286, 0.105, 0]])

# open image
I = io.imread(File)

# perform color deconvolution
Unmixed = ColorDeconvolution(I, W)

# view output - check if coherent
plt.figure
plt.subplot(2, 2, 1)
plt.imshow(I)
plt.title('Color Image')
plt.subplot(2, 2, 2)
plt.imshow(Unmixed.Stains[:, :, 0], cmap=cm.Greys_r)
plt.title('Hematoxylin')
plt.subplot(2, 2, 3)
plt.imshow(Unmixed.Stains[:, :, 1], cmap=cm.Greys_r)
plt.title('Eosin')
plt.subplot(2, 2, 4)
plt.imshow(Unmixed.Stains[:, :, 2], cmap=cm.Greys_r)
plt.title('Complement')
