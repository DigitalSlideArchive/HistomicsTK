import matplotlib.pyplot as plt
import numpy as np
import os

# Define Romanesco globals for the style checker
inputImageFile = inputImageFile  # noqa
_tempdir = _tempdir   # noqa


def ComplementStainMatrix(W):
    """
    Used to fill out empty columns of a stain matrix for use with
    ColorDeconvolution.
    """
    Complemented = W

    if (W[0, 0]**2 + W[0, 1]**2) > 1:
        Complemented[0, 2] = 0
    else:
        Complemented[0, 2] = (1 - (W[0, 0]**2 + W[0, 1]**2))**0.5

    if (W[1, 0]**2 + W[1, 1]**2) > 1:
        Complemented[1, 2] = 0
    else:
        Complemented[1, 2] = (1 - (W[1, 0]**2 + W[1, 1]**2))**0.5

    if (W[2, 0]**2 + W[2, 1]**2) > 1:
        Complemented[2, 2] = 0
    else:
        Complemented[2, 2] = (1 - (W[2, 0]**2 + W[2, 1]**2))**0.5

    Complemented[:, 2] = Complemented[:, 2] / np.linalg.norm(Complemented[:, 2])

    return Complemented


def OpticalDensityFwd(I):
    """
    Transforms input RGB image "I" into optical density space for color
    deconvolution.

    inputs:
        I -
    outputs:
    """
    return -(255 * np.log(I / 255)) / np.log(255)


def OpticalDensityInv(I):
    """
    Transforms input RGB image "I" into optical density space for color
    deconvolution.

    inputs:
        I -
    outputs:
    """
    return np.exp(-(I - 255) * np.log(255) / 255)


def ColorDeconvolution(I, W):
    """
    Implements color deconvolution. The input image "I" consisting of RGB values
    is first transformed into optical density space, and then projected onto the
    stain vectors in the columns of "W".

    example H&E matrix
    W =array([[0.650, 0.072, 0],
              [0.704, 0.990, 0],
              [0.286, 0.105, 0]])
    inputs:
        I -
        W -
    outputs:
    """
    # complement stain matrix if needed
    if np.linalg.norm(W[:, 2]) <= 1e-16:
        W = ComplementStainMatrix(W)

    print W
    print W.shape

    # normalize stains to unit-norm
    for i in range(W.shape[1]):
        Norm = np.linalg.norm(W[:, i])
        if Norm >= 1e-16:
            W[:, i] /= Norm

    # invert stain matrix
    Q = np.linalg.inv(W)

    # transform 3D input image to 2D RGB matrix format
    m = I.shape[0]
    n = I.shape[1]
    if I.shape[2] == 4:
        I = I[:, :, (0, 1, 2)]
    I = np.reshape(I, (m * n, 3))

    # transform input RGB to optical density values and deconvolve, tfm back to
    # RGB
    I = I.astype(dtype=np.float32)
    I[I == 0] = 1e-16
    ODfwd = OpticalDensityFwd(I)
    ODdeconv = np.dot(ODfwd, np.transpose(Q))
    ODinv = OpticalDensityInv(ODdeconv)

    # reshape output, transform type
    I = np.reshape(ODinv, (m, n, 3))
    I[I > 255] = 255
    I = I.astype(np.uint8)

    return I

print('starting')

print(inputImageFile)

outputImageFile = os.path.join(_tempdir,
                               'out_' + os.path.split(inputImageFile)[1])
print(outputImageFile)

inputImage = plt.imread(inputImageFile)

# outputImage = inputImage
W = np.array([[0.650, 0.072, 0], [0.704, 0.990, 0], [0.286, 0.105, 0]])
outputImage = ColorDeconvolution(inputImage, W)

plt.imsave(outputImageFile, outputImage)
