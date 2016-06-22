# Tests ReinhardNorm.py. Normalizes an input image to a standard image in LAB
# color space. The input image, standard image, and normalized output image are
# displayed side-by-side.
# Also tests helper functions:
# RudermanLABFwd.py
# RudermanLABInv.py

import matplotlib.pyplot as plt
import numpy
import openslide
import histomicstk as htk

# Define input slides, subregions
StandardFile = '/Users/lcoop22/Desktop/ExampleData/' \
               'TCGA-02-0003-01Z-00-DX2.c7652d8d-d78f-49ae-8259-bb2bdb5d50fb' \
               '.svs'
StandardX = 22000
StandardY = 12800
# InputFile = '/Users/lcoop22/Desktop/ExampleData/' \
#             'TCGA-02-0009-01Z-00-DX3.BAA1276B-E4D7-43D4-BDDF-807532462518.svs'
# InputX = 64500
# InputY = 22900
# InputFile = '/Users/lcoop22/Desktop/ExampleData/' \
#             'TCGA-02-0010-01Z-00-DX2.5334831b-8e1f-4b61-bbf6-0f6e950a1b2f.svs'
# InputX1 = 32400
# InputY1 = 19000
# InputX2 = 35000
# InputY2 = 19000
InputFile = '/Users/lcoop22/Desktop/ExampleData/' \
            'TCGA-02-0010-01Z-00-DX4.07de2e55-a8fe-40ee-9e98-bcb78050b9f7.svs'
InputX1 = 13900
InputY1 = 15000
InputX2 = 6500
InputY2 = 15000
T = 2048

# sample from standard slide to generate target parameters
TargetStats = htk.ReinhardSample(StandardFile, 20, 0.001, 128)

# #sample from input slide to generate input parameters
InputStats = htk.ReinhardSample(InputFile, 20, 0.001, 128)

# open slides and import regions
StandardSlide = openslide.OpenSlide(StandardFile)
StandardTile = StandardSlide.read_region((StandardX, StandardY), 0, (T, T))
InputSlide = openslide.OpenSlide(InputFile)
InputTile1 = InputSlide.read_region((InputX1, InputY1), 0, (T, T))
InputTile2 = InputSlide.read_region((InputX2, InputY2), 0, (T, T))

# strip out alpha channels and convert to numpy arrays
InputTile1 = numpy.asarray(InputTile1)
InputTile1 = InputTile1[:, :, :3]
InputTile2 = numpy.asarray(InputTile2)
InputTile2 = InputTile2[:, :, :3]
StandardTile = numpy.asarray(StandardTile)
StandardTile = StandardTile[:, :, :3]

# normalize inputs to standard (global)
Input1Standard = htk.ReinhardNorm(InputTile1, TargetStats.Mu, TargetStats.Sigma,
                                  InputStats.Mu, InputStats.Sigma)
Input2Standard = htk.ReinhardNorm(InputTile2, TargetStats.Mu, TargetStats.Sigma,
                                  InputStats.Mu, InputStats.Sigma)

# normalize input to standard (local)
lInput1Standard = htk.ReinhardNorm(InputTile1, TargetStats.Mu,
                                   TargetStats.Sigma)
lInput2Standard = htk.ReinhardNorm(InputTile2, TargetStats.Mu,
                                   TargetStats.Sigma)

# view output - check if coherent
plt.figure
plt.subplot(2, 4, 1)
plt.imshow(StandardTile)
plt.title('Tile - "standardization" slide')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(InputTile1)
plt.title('Tile 1 - input slide')
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(InputTile2)
plt.title('Tile 2 - input slide')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(lInput1Standard)
plt.title('Tile 1 - input slide - local norm')
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(lInput2Standard)
plt.title('Tile 2 - input slide - local norm')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(Input1Standard)
plt.title('Tile 1 - input slide - global norm')
plt.axis('off')

plt.subplot(2, 4, 8)
plt.imshow(Input2Standard)
plt.title('Tile 2 - input slide - global norm')
plt.axis('off')
