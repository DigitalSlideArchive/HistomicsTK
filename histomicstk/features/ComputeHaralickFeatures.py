import numpy as np
import pandas as pd
from skimage.measure import regionprops
from .graycomatrixext import graycomatrixext
from .graycomatrixext import _default_num_levels


def ComputeHaralickFeatures(im_label, im_intensity, offsets=None,
                            num_levels=None, gray_limits=None, rprops=None):
    """
    Calculates 26 Haralick features from an intensity image with the labels.

    Parameters
    ----------
    im_label : array_like
        An ND labeled mask image wherein intensity of a pixel is the ID of the
        object it belongs to. Non-zero values are considered to be foreground
        objects.

    im_intensity : array_like
        An ND single channel intensity image

    offsets : array_like, optional
        A (num_offsets, num_image_dims) array of offset vectors
        specifying the distance between the pixel-of-interest and
        its neighbor. Note that the first dimension corresponds to
        the rows.

        Because this offset is often expressed as an angle, the
        following table lists the offset values that specify common
        angles for a 2D image, given the pixel distance D.

        AngleXY  |  OFFSET
        -------  |  ------
        0        |  [0 D]
        45       |  [-D D]
        90       |  [-D 0]
        135      |  [-D -D]

        Default
            [1] for 1D,
            [[1, 0], [0, 1]] for 2D,
            [[1, 0, 0], [0, 1, 0], [0, 0, 1] for 3D and so on

    num_levels : unsigned int, optional
        An integer specifying the number of gray levels For example, if
        `NumLevels` is 8,  the intensity values of the input image are
        scaled so they are integers between 1 and 8.  The number of gray
        levels determines the size of the gray-level co-occurrence matrix.

        Default: 2 for binary/logical image, 32 for numeric image

    gray_limits : array_like, optional
        A two-element array specifying the desired input intensity range.
        Intensity values in the input image will be clipped into this range.

        Default: [0, 1] for boolean-valued image, [0, 255] for integer-valued
        image, and [0.0, 1.0] for-real valued image

    Returns
    -------
    fdata: pandas.DataFrame
        A pandas dataframe containing the haralick features.

    Notes
    -----
    List of haralick features computed by this function:

    Haralick.ASM.Mean : float
        Mean of angular second moment of object pixels.

    Haralick.Contrast.Mean : float
        Mean of contrast of object pixels.

    Haralick.Correlation.Mean : float
        Mean of correlation of object pixels.

    Haralick.SumofSquar.Mean : float
        Mean of sum of squares of object pixels.

    Haralick.IDM.Mean : float
        Mean of inverse difference moment of object pixels.

    Haralick.SumAverage.Mean : float
        Mean of sum average of object pixels.

    Haralick.SumVariance.Mean : float
        Mean of sum variance of object pixels.

    Haralick.SumEntropy.Mean : float
        Mean of sum entropy of object pixels.

    Haralick.Entropy.Mean : float
        Mean of entropy of object pixels.

    Haralick.Variance.Mean : float
        Mean of variance of object pixels.

    Haralick.DifferenceEntropy.Mean : float
        Mean of differnce entropy of object pixels.

    Haralick.IMC1.Mean : float
        Mean of the first information measures of correlation of object
        pixels.

    Haralick.IMC2.Mean : float
        Mean of the second information measures of correlation of object
        pixels.

    Haralick.ASM.Range : float
        Range of angular second moment of object pixels.

    Haralick.Contrast.Range : float
        Range of contrast of object pixels.

    Haralick.Correlation.Range : float
        Range of correlation of object pixels.

    Haralick.SumofSquar.Range : float
        Range of sum of squares of object pixels.

    Haralick.IDM.Range : float
        Range of inverse difference moment of object pixels.

    Haralick.SumAverage.Range : float
        Range of sum avarage of object pixels.

    Haralick.SumVariance.Range : float
        Range of sum variance of object pixels.

    Haralick.SumEntropy.Range : float
        Range of sum entropy of object pixels.

    Haralick.Entropy.Range : float
        Range of entropy of object pixels.

    Haralick.Variance.Range : float
        Range of variance of object pixels.

    Haralick.DifferenceEntropy.Range : float
        Range of differnece entropy of object pixels.

    Haralick.IMC1.Range : float
        Range the first information measures of correlation of object
        pixels.

    Haralick.IMC2.Range : float
        Range the second information measures of correlation of object
        pixels.

    References
    ----------
    .. [1] Haralick, et al. "Textural features for image classification,"
    IEEE Transactions on Systems, Man, and Cybernatics, vol. 6, pp: 610-621,
    1973.
    .. [2] Luis Pedro Coelho. "Mahotas: Open source software for scriptable
    computer vision," Journal of Open Research Software, vol 1, 2013.
    """

    # List of feature names
    feature_list = [
        'Haralick.ASM.Mean',
        'Haralick.Contrast.Mean',
        'Haralick.Correlation.Mean',
        'Haralick.SumofSquar.Mean',
        'Haralick.IDM.Mean',
        'Haralick.SumAverage.Mean',
        'Haralick.SumVariance.Mean',
        'Haralick.SumEntropy.Mean',
        'Haralick.Entropy.Mean',
        'Haralick.Variance.Mean',
        'Haralick.DifferenceEntropy.Mean',
        'Haralick.IMC1.Mean',
        'Haralick.IMC2.Mean',
        'Haralick.ASM.Range',
        'Haralick.Contrast.Range',
        'Haralick.Correlation.Range',
        'Haralick.SumofSquar.Range',
        'Haralick.IDM.Range',
        'Haralick.SumAverage.Range',
        'Haralick.SumVariance.Range',
        'Haralick.SumEntropy.Range',
        'Haralick.Entropy.Range',
        'Haralick.Variance.Range',
        'Haralick.DifferenceEntropy.Range',
        'Haralick.IMC1.Range',
        'Haralick.IMC2.Range',
    ]

    # num_levels
    if num_levels is None:
        num_levels = _default_num_levels(im_intensity)

    # check for consistent shapes between 'I' and 'Label'
    if im_intensity.shape != im_label.shape:
        raise ValueError("Inputs 'I' and 'Label' must have same shape")

    num_dims = len(im_intensity.shape)

    # offsets
    if offsets is None:
        # set default offset value
        offsets = np.identity(num_dims)
    else:
        # check sanity
        if offsets.shape[1] != num_dims:
            raise ValueError(
                'Dimension mismatch between input image and offsets'
            )

    num_offsets = offsets.shape[0]

    # compute object properties if not provided
    if rprops is None:
        rprops = regionprops(im_label)

    # create pandas data frame containing the features for each object
    numFeatures = len(feature_list)
    numLabels = len(rprops)
    fdata = pd.DataFrame(np.zeros((numLabels, numFeatures)),
                         columns=feature_list)

    for i in range(numLabels):

        # get bounds of an intensity image
        minr, minc, maxr, maxc = rprops[i].bbox

        # grab nucleus mask
        subImage = im_intensity[minr:maxr+1, minc:maxc+1].astype(np.uint8)

        # gets GLCM or gray-tone spatial dependence matrix
        arrayGLCM = graycomatrixext(subImage, offsets=offsets,
                                    num_levels=num_levels,
                                    gray_limits=gray_limits,
                                    symmetric=True, normed=True)

        # List of local feature names
        local_feature_list = [
            'Haralick.ASM',
            'Haralick.Contrast',
            'Haralick.Correlation',
            'Haralick.SumofSquar',
            'Haralick.IDM',
            'Haralick.SumAverage',
            'Haralick.SumVariance',
            'Haralick.SumEntropy',
            'Haralick.Entropy',
            'Haralick.Variance',
            'Haralick.DifferenceEntropy',
            'Haralick.IMC1',
            'Haralick.IMC2',
        ]

        ldata = pd.DataFrame(np.zeros((num_offsets, len(local_feature_list))),
                             columns=local_feature_list)

        for r in range(num_offsets):

            nGLCM = arrayGLCM[:, :, r]

            # get marginal-probability matrix summing the rows
            px = np.sum(nGLCM, axis=1)
            py = np.sum(nGLCM, axis=0)

            # initialize marginal-probability matrix sets
            # summing normalizedGLCM such that i+j = k or i-j = k
            pxPlusy = np.zeros(2*num_levels-1)
            pxMinusy = np.zeros(num_levels)

            # arbitarily small positive constant to avoid log 0
            e = 0.00001
            for n in range(0, num_levels):
                for m in range(0, num_levels):

                    # gets marginal-probability matrix
                    pxPlusy[n+m] = pxPlusy[n+m] + nGLCM[n, m]
                    pxMinusy[abs(n-m)] = pxMinusy[abs(n-m)] + nGLCM[n, m]

            # computes angular second moment
            ldata.at[r, 'Haralick.ASM'] = np.sum(np.square(nGLCM))

            # computes contrast
            n_Minus = np.arange(num_levels)
            ldata.at[r, 'Haralick.Contrast'] = \
                np.dot(np.square(n_Minus), pxMinusy)

            # computes correlation
            # gets weighted mean and standard deviation of px and py
            meanx = np.dot(n_Minus, px)
            variance = np.dot(px, np.square(n_Minus)) - np.square(meanx)
            nGLCMr = np.ravel(nGLCM)
            x, y = np.mgrid[0:num_levels, 0:num_levels]
            xy = x*y
            ldata.at[r, 'Haralick.Correlation'] = \
                (np.dot(np.ravel(xy), nGLCMr) - np.square(meanx)) / variance

            # computes sum of squares : variance
            ldata.at[r, 'Haralick.SumofSquar'] = variance

            # computes inverse difference moment
            xy_IDM = 1. / (1+np.square(x-y))
            ldata.at[r, 'Haralick.IDM'] = \
                np.dot(np.ravel(xy_IDM), nGLCMr)

            # computes sum average
            n_Plus = np.arange(2*num_levels-1)
            ldata.at[r, 'Haralick.SumAverage'] = \
                np.dot(n_Plus, pxPlusy)

            # computes sum variance
            # [1] uses sum entropy, but we use sum average
            ldata.at[r, 'Haralick.SumVariance'] = \
                np.dot(np.square(n_Plus), pxPlusy) - \
                np.square(ldata.at[r, 'Haralick.SumAverage'])

            # computes sum entropy
            ldata.at[r, 'Haralick.SumEntropy'] = \
                -np.dot(pxPlusy, np.log2(pxPlusy+e))

            # computes entropy
            ldata.at[r, 'Haralick.Entropy'] = \
                -np.dot(nGLCMr, np.log2(nGLCMr+e))

            # computes variance px-y
            ldata.at[r, 'Haralick.Variance'] = np.var(pxMinusy)

            # computes difference entropy px-y
            ldata.at[r, 'Haralick.DifferenceEntropy'] = \
                -np.dot(pxMinusy, np.log2(pxMinusy+e))

            # computes information measures of correlation
            # gets entropies of px and py
            HX = -np.dot(px, np.log2(px+e))
            HY = -np.dot(py, np.log2(py+e))
            HXY = ldata.at[r, 'Haralick.Entropy']
            pxy_ij = np.outer(px, py)
            pxy_ijr = np.ravel(pxy_ij)
            HXY1 = -np.dot(nGLCMr, np.log2(pxy_ijr+e))
            HXY2 = -np.dot(pxy_ijr, np.log2(pxy_ijr+e))
            ldata.at[r, 'Haralick.IMC1'] = (HXY-HXY1)/max(HX, HY)

            # computes information measures of correlation
            ldata.at[r, 'Haralick.IMC2'] = \
                np.sqrt(1 - np.exp(-2.0*(HXY2-HXY)))

        fdata.at[i, 'Haralick.ASM.Mean'] = \
            np.mean(ldata.loc[:, 'Haralick.ASM'])
        fdata.at[i, 'Haralick.Contrast.Mean'] = \
            np.mean(ldata.loc[:, 'Haralick.Contrast'])
        fdata.at[i, 'Haralick.Correlation.Mean'] = \
            np.mean(ldata.loc[:, 'Haralick.Correlation'])
        fdata.at[i, 'Haralick.SumofSquar.Mean'] = \
            np.mean(ldata.loc[:, 'Haralick.SumofSquar'])
        fdata.at[i, 'Haralick.IDM.Mean'] = \
            np.mean(ldata.loc[:, 'Haralick.IDM'])
        fdata.at[i, 'Haralick.SumAverage.Mean'] = \
            np.mean(ldata.loc[:, 'Haralick.SumAverage'])
        fdata.at[i, 'Haralick.SumVariance.Mean'] = \
            np.mean(ldata.loc[:, 'Haralick.SumVariance'])
        fdata.at[i, 'Haralick.SumEntropy.Mean'] = \
            np.mean(ldata.loc[:, 'Haralick.SumEntropy'])
        fdata.at[i, 'Haralick.Entropy.Mean'] = \
            np.mean(ldata.loc[:, 'Haralick.Entropy'])
        fdata.at[i, 'Haralick.Variance.Mean'] = \
            np.mean(ldata.loc[:, 'Haralick.Variance'])
        fdata.at[i, 'Haralick.DifferenceEntropy.Mean'] = \
            np.mean(ldata.loc[:, 'Haralick.DifferenceEntropy'])
        fdata.at[i, 'Haralick.IMC1.Mean'] = \
            np.mean(ldata.loc[:, 'Haralick.IMC1'])
        fdata.at[i, 'Haralick.IMC2.Mean'] = \
            np.mean(ldata.loc[:, 'Haralick.IMC2'])
        fdata.at[i, 'Haralick.ASM.Range'] = \
            np.ptp(ldata.loc[:, 'Haralick.ASM'])
        fdata.at[i, 'Haralick.Contrast.Range'] = \
            np.ptp(ldata.loc[:, 'Haralick.Contrast'])
        fdata.at[i, 'Haralick.Correlation.Range'] = \
            np.ptp(ldata.loc[:, 'Haralick.Correlation'])
        fdata.at[i, 'Haralick.SumofSquar.Range'] = \
            np.ptp(ldata.loc[:, 'Haralick.SumofSquar'])
        fdata.at[i, 'Haralick.IDM.Range'] = \
            np.ptp(ldata.loc[:, 'Haralick.IDM'])
        fdata.at[i, 'Haralick.SumAverage.Range'] = \
            np.ptp(ldata.loc[:, 'Haralick.SumAverage'])
        fdata.at[i, 'Haralick.SumVariance.Range'] = \
            np.ptp(ldata.loc[:, 'Haralick.SumVariance'])
        fdata.at[i, 'Haralick.SumEntropy.Range'] = \
            np.ptp(ldata.loc[:, 'Haralick.SumEntropy'])
        fdata.at[i, 'Haralick.Entropy.Range'] = \
            np.ptp(ldata.loc[:, 'Haralick.Entropy'])
        fdata.at[i, 'Haralick.Variance.Range'] = \
            np.ptp(ldata.loc[:, 'Haralick.Variance'])
        fdata.at[i, 'Haralick.DifferenceEntropy.Range'] = \
            np.ptp(ldata.loc[:, 'Haralick.DifferenceEntropy'])
        fdata.at[i, 'Haralick.IMC1.Range'] = \
            np.ptp(ldata.loc[:, 'Haralick.IMC1'])
        fdata.at[i, 'Haralick.IMC2.Range'] = \
            np.ptp(ldata.loc[:, 'Haralick.IMC2'])

    return fdata
