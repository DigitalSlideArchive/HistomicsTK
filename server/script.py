import matplotlib.pyplot as plt
import numpy as np
import os

import histomicstk as htk

print('starting')

print(inputImageFile)

outputImageFile = os.path.join(_tempdir,
                               'out_'+ os.path.split(inputImageFile)[1])
print(outputImageFile)

inputImage = plt.imread(inputImageFile)

W = np.array([[0.650, 0.072, 0], [0.704, 0.990, 0], [0.286, 0.105, 0]])
outputImage = htk.ColorDeconvolution(inputImage, W)

plt.imsave(outputImageFile, outputImage)


