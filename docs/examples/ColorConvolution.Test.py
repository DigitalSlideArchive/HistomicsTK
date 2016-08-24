# Tests ColorConvolution.py. Uses a standard H&E calibration matrix to
# deconvolve a test image into hematoxylin and eosin stains, uses these stains
# to recompose the color image and each stain, and then displays the original
# and recomposed images side-by-side.
# Also tests helper functions:
# ComplementStainMatrix.py
# rgb_to_od.py
# od_to_rgb.py

import matplotlib.pyplot as plt
import numpy
import skimage.io as io

# Define globals for the style checker
# TODO: Ensure these are actually provided
ColorDeconvolution = ColorDeconvolution  # noqa
ColorConvolution = ColorConvolution   # noqa

# Define input image and H&E color matrix
File = './B.png'
W = numpy.array([[0.650, 0.072, 0],
                 [0.704, 0.990, 0],
                 [0.286, 0.105, 0]])

# open image
I = io.imread(File)

# perform color deconvolution
Unmixed = ColorDeconvolution(I, W)

# reconvolve color image
Remixed = ColorConvolution(Unmixed.Stains, W)

# color image of hematoxylin stain
WH = W.copy()
WH[:, 1] = 0
WH[:, 2] = 0
Hematoxylin = ColorConvolution(Unmixed.Stains, WH)

# color image of eosin stain
WE = W.copy()
WE[:, 0] = 0
WE[:, 2] = 0
Eosin = ColorConvolution(Unmixed.Stains, WE)

# view output - check if coherent
plt.figure
plt.subplot(2, 2, 1)
plt.imshow(I)
plt.title('Color Image')
plt.subplot(2, 2, 2)
plt.imshow(Remixed)
plt.title('Remixed - Uncomplemented Matrix')
plt.subplot(2, 2, 3)
plt.imshow(Hematoxylin)
plt.title('Remixed - Hematoxylin')
plt.subplot(2, 2, 4)
plt.imshow(Eosin)
plt.title('Remixed - Eosin')
