import skimage.feature
import numpy as np


def FeatureExtractionHaralick(I, Dst=1, Ng=256):
    """
    Calculates 26 Haralick features from an image
    Parameters
    ----------
    I : array_like
        An RGB image where in each channel contains image of one stain
    Dst : Distance of neighboring pixels in an image. Default value = 1.
    Ng : Number of gray level co-occurance matrix. Default value = 256.
    Returns
    -------
    H : array_like
        A matrix containing mean and ranges for 26 features
    References
    ----------
    .. [1] Haralick, et al. "Textural features for image classification,"
    IEEE Transactions on Systems, Man, and Cybernatics, vol. 6, pp: 610-621,
    1973.
    .. [2] Luis Pedro Coelho. "Mahotas: Open source software for scriptable
    computer vision," Journal of Open Research Software, vol 1, 2013.
    """
    # sets the number of angles: 0 45 90 135. this must not be modified.
    NumberofAngles = 4

    # gets GLCM or gray-tone spatial dependence matrix
    P_ij = skimage.feature.greycomatrix(
        I, [Dst], [0, np.pi/4, np.pi/2, 3*np.pi/4],
        symmetric=True, levels=Ng
    )
    # initialize a feature set for 13 features
    f = np.zeros((13, NumberofAngles))

    for r in range(0, NumberofAngles):
        # gets sum of array for each angle
        R = np.sum(P_ij[:, :, 0, r], dtype=np.float)
        # gets normalized gray-tone spatial dependence matrix
        p_ij = P_ij[:, :, 0, r]/R
        # gets marginal-probability matrix sets summing the rows of p_ij
        px = np.sum(p_ij, axis=1)
        # initialize marginal-probability matrix sets summing p_ij
        # that i+j = k or i-j = k
        pxPlusy = np.zeros(2*Ng-1)
        pxMinusy = np.zeros(Ng)
        # arbitarily small positive constant
        # e = 0.00001
        for i in range(0, Ng):
            for j in range(0, Ng):
                # gets marginal-probability matrix
                pxPlusy[i+j] = pxPlusy[i+j] + p_ij[i, j]
                pxMinusy[abs(i-j)] = pxMinusy[abs(i-j)] + p_ij[i, j]
        # f0: computes angular second moment
        f[0, r] = np.sum(p_ij**2)
        # f1: computes constast
        n_Minus = np.arange(Ng)
        f[1, r] = np.dot(n_Minus**2, pxMinusy)
        # f2: computes correlation
        # gets weighted mean and weighted standard deviation of px and py
        meanx = np.dot(n_Minus, px)
        variance = np.dot(px, n_Minus**2) - meanx**2
        p_ijr = np.ravel(p_ij)
        i, j = np.mgrid[0:Ng, 0:Ng]
        ij = i*j
        f[2, r] = (np.dot(np.ravel(ij), p_ijr) - meanx**2) / variance
        # f3: computes sum of squares : variance
        f[3, r] = variance
        # f4: computes inverse difference moment
        ij_IDM = 1. / (1+(i-j)**2)
        f[4, r] = np.dot(np.ravel(ij_IDM), p_ijr)
        f[5, r] = 0
        f[6, r] = 0
        f[7, r] = 0
        f[8, r] = 0
        f[9, r] = 0
        f[10, r] = 0
        f[11, r] = 0
        f[12, r] = 0
    # computes means and ranges of the 13 features with 4 angles
    # and store it to H
    H = np.zeros(26)
    H[:13] = np.mean(f, axis=1)
    H[13:26] = np.ptp(f, axis=1)

    return H
