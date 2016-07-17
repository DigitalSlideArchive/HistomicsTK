import numpy as np
import pandas as pd
import skimage.feature
from skimage.measure import regionprops


def ComputeHaralickFeatures(I, Label, Offsets=[[0, 1]], NumLevels=256,
                            MaxGray=255, MinGray=0):
    """
    Calculates 26 Haralick features from an intensity image with the labels.

    Parameters
    ----------
    I : array_like
        M x N intensity image.
    Label : array_like
        M x N label image.
    Offsets : array_like
        Specifies common angles for a 2D image, given the pixel distacne D.
        Default value = [[0, 1]]
        AngleXY     OFFSET
        -------     ------
        0           [0 D]
        45          [-D D]
        90          [-D 0]
        135         [-D -D]
    NumLevels : integer
        Specifies the number of gray level. NumLevels determines the size of
        the gray-level co-occurrence matrix.
        Default value = 256.
    MaxGray : integer
        Specifies a maximum grayscale value in I. Defalut value = 255
    MinGray : integer
        Specifies a minimum grayscale value in I. Defalut value = 0

    Returns
    -------
    df : 2-dimensional labeled data structure, float64
        Pandas data frame.

    References
    ----------
    .. [1] Haralick, et al. "Textural features for image classification,"
    IEEE Transactions on Systems, Man, and Cybernatics, vol. 6, pp: 610-621,
    1973.
    .. [2] Luis Pedro Coelho. "Mahotas: Open source software for scriptable
    computer vision," Journal of Open Research Software, vol 1, 2013.
    """

    # check for consistent shapes between 'I' and 'Label'
    if I.shape != Label.shape:
        raise ValueError("Inputs 'I' and 'Label' must have same shape")

    # determine if image is grayscale or RGB
    if len(I.shape) != 2:  # color image
        raise ValueError("Inputs 'I' should be a grayscale image")

    # check for Offset
    arrayOffset = np.asarray(Offsets)
    offsetShape = arrayOffset.shape
    if (len(offsetShape) != 2):
        raise ValueError("Shape of Offset should be an numOffsets by 2")

    # restict the range of intensity from MinGray to MaxGray
    I = np.clip(I, MinGray, MaxGray)

    # initialize feature names
    featureList = [
        'ASM', 'Contrast', 'Correlation', 'SumofSquar',
        'IDM', 'SumAverage', 'SumVariance', 'SumEntropy', 'Entropy',
        'Variance', 'DifferenceEntropy', 'IMC1', 'IMC2'
    ]

    # initialize panda dataframe
    columnList = []
    for i in range(0, len(featureList)):
        columnList.append(featureList[i] + 'Mean')
        columnList.append(featureList[i] + 'Range')
    df = pd.DataFrame(columns=columnList)

    # gets total number of regions
    numofLabels = Label.max()

    # decodes angels and distances
    numOffsets = offsetShape[0]
    listAngles = []
    listDistances = []

    # compute angles from cartesian coordinates
    arrayAngles = np.arctan2(arrayOffset[:, 0], arrayOffset[:, 1])
    for i in range(0, len(arrayAngles)):
        if arrayAngles[i] == 0:
            listDistances = np.append(listDistances, abs(arrayOffset[i, 1]))
        elif arrayAngles[i] == np.pi/2:
            listDistances = np.append(listDistances, abs(arrayOffset[i, 0]))
        elif arrayAngles[i] == np.pi/4:
            listDistances = np.append(listDistances, abs(arrayOffset[i, 1]))
        elif arrayAngles[i] == 3*np.pi/4:
            listDistances = np.append(listDistances, abs(arrayOffset[i, 0]))
        else:
            raise ValueError("Current Offset format is not availabe.")

    listAngles = arrayAngles.tolist()

    # initialize sub panda dataframe
    subDataframe = np.zeros((numofLabels, len(featureList)*2))

    # extract feature information
    for region in regionprops(Label):
        # get bounds of an intensity image
        box = region.bbox
        # grab nucleus mask
        subImage = I[box[0]:box[2], box[1]:box[3]].astype(np.uint8)

        # gets GLCM or gray-tone spatial dependence matrix
        arrayGLCM = skimage.feature.greycomatrix(
            subImage, listDistances, listAngles,
            symmetric=True, levels=NumLevels
        )
        # gets size x and y of GLCM
        sizeX = arrayGLCM.shape[0]
        sizeY = arrayGLCM.shape[1]
        offsetsGLCM = np.zeros((sizeX, sizeY, numOffsets))
        # initialize a feature set for 13 features
        f = np.zeros((13, numOffsets))
        # initialize H for 26 features
        H = np.zeros(26)
        for r in range(0, numOffsets):
            # get each offset GLCM
            offsetsGLCM[:, :, r] = arrayGLCM[:, :, r, r]
            # normalize GLCM
            R = np.sum(offsetsGLCM[:, :, r], dtype=np.float)
            nGLCM = offsetsGLCM[:, :, r]/R
            # get marginal-probability matrix summing the rows
            px = np.sum(nGLCM, axis=1)
            py = np.sum(nGLCM, axis=0)
            # initialize marginal-probability matrix sets
            # summing normalizedGLCM such that i+j = k or i-j = k
            pxPlusy = np.zeros(2*NumLevels-1)
            pxMinusy = np.zeros(NumLevels)
            # arbitarily small positive constant to avoid log 0
            e = 0.00001
            for i in range(0, NumLevels):
                for j in range(0, NumLevels):
                    # gets marginal-probability matrix
                    pxPlusy[i+j] = pxPlusy[i+j] + nGLCM[i, j]
                    pxMinusy[abs(i-j)] = pxMinusy[abs(i-j)] + nGLCM[i, j]
            # f0: computes angular second moment
            f[0, r] = np.sum(np.square(nGLCM))
            # f1: computes contrast
            n_Minus = np.arange(NumLevels)
            f[1, r] = np.dot(np.square(n_Minus), pxMinusy)
            # f2: computes correlation
            # gets weighted mean and standard deviation of px and py
            meanx = np.dot(n_Minus, px)
            variance = np.dot(px, np.square(n_Minus)) - np.square(meanx)
            nGLCMr = np.ravel(nGLCM)
            i, j = np.mgrid[0:NumLevels, 0:NumLevels]
            ij = i*j
            f[2, r] = (np.dot(np.ravel(ij), nGLCMr) - np.square(meanx)) / \
                variance
            # f3: computes sum of squares : variance
            f[3, r] = variance
            # f4: computes inverse difference moment
            ij_IDM = 1. / (1+np.square(i-j))
            f[4, r] = np.dot(np.ravel(ij_IDM), nGLCMr)
            # f5: computes sum average
            n_Plus = np.arange(2*NumLevels-1)
            f[5, r] = np.dot(n_Plus, pxPlusy)
            # f6: computes sum variance
            # [1] uses sum entropy, but we use sum average
            f[6, r] = np.dot(np.square(n_Plus), pxPlusy) - \
                np.square(f[5, r])
            # f7: computes sum entropy
            f[7, r] = -np.dot(pxPlusy, np.log2(pxPlusy+e))
            # f8: computes entropy
            f[8, r] = -np.dot(nGLCMr, np.log2(nGLCMr+e))
            # f9: computes variance px-y
            f[9, r] = np.var(pxMinusy)
            # f10: computes difference entropy px-y
            f[10, r] = -np.dot(pxMinusy, np.log2(pxMinusy+e))
            # f11: computes information measures of correlation
            # gets entropies of px and py
            HX = -np.dot(px, np.log2(px+e))
            HY = -np.dot(py, np.log2(py+e))
            HXY = f[8, r]
            pxy_ij = np.outer(px, py)
            pxy_ijr = np.ravel(pxy_ij)
            HXY1 = -np.dot(nGLCMr, np.log2(pxy_ijr+e))
            HXY2 = -np.dot(pxy_ijr, np.log2(pxy_ijr+e))
            f[11, r] = (HXY-HXY1)/max(HX, HY)
            # f12: computes information measures of correlation
            f[12, r] = np.sqrt(1 - np.exp(-2.0*(HXY2-HXY)))
        # computes means and ranges of the features
        H[:13] = np.mean(f, axis=1)
        H[13:26] = np.ptp(f, axis=1)

        subDataframe[region.label-1, :] = H

    for i in range(0, len(featureList)):
        df[featureList[i] + 'Mean'] = subDataframe[:, i]
        df[featureList[i] + 'Range'] = subDataframe[:, i+len(featureList)]

    return df
