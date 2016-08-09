import numpy as np
import pandas as pd
from skimage.measure import regionprops
from graycomatrixext import graycomatrixext


def ComputeHaralickFeatures(im_label, im_intensity, offsets=None, num_levels=8,
                            gray_limits=[0, 255], rprops=None):
    """
    Calculates 26 Haralick features from an intensity image with the labels.

    Parameters
    ----------
    im_label : array_like
        A labeled mask image wherein intensity of a pixel is the ID of the
        object it belongs to. Non-zero values are considered to be foreground
        objects.

    im_intensity : array_like
        Intensity image

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
        `num_levels` is 8,  the intensity values of the input image are
        scaled so they are integers between 1 and 8.  The number of gray
        levels determines the size of the gray-level co-occurrence matrix.

        Default: 8 for numeric image, 2 for binary/logical image

    gray_limits : array_like, optional
        A two-element array specifying the desired input intensity range.
        Intensity values in the input image will be clipped into this range.

        Default: [0, 1] for binary/logical image, [0, 255] for numeric image

    Returns
    -------
    fdata: pandas.DataFrame
        A pandas dataframe containing the haralick features.

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

    # compute object properties if not provided
    if rprops is None:
        rprops = regionprops(im_label)

    # create pandas data frame containing the features for each object
    numFeatures = len(feature_list)
    numLabels = len(rprops)
    fdata = pd.DataFrame(np.zeros((numLabels, numFeatures)),
                         columns=feature_list)

    # check for consistent shapes between 'I' and 'Label'
    if im_intensity.shape != im_label.shape:
        raise ValueError("Inputs 'I' and 'Label' must have same shape")

    # determine if image is grayscale or RGB
    if len(im_intensity.shape) != 2:  # color image
        raise ValueError("Inputs 'I' should be a grayscale image")

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
        num_dims = len(subImage.shape)

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

            # f0: computes angular second moment
            ldata.at[r, 'Haralick.ASM'] = np.sum(np.square(nGLCM))

            # f1: computes contrast
            n_Minus = np.arange(num_levels)
            ldata.at[r, 'Haralick.Contrast'] = \
                np.dot(np.square(n_Minus), pxMinusy)

            # f2: computes correlation
            # gets weighted mean and standard deviation of px and py
            meanx = np.dot(n_Minus, px)
            variance = np.dot(px, np.square(n_Minus)) - np.square(meanx)
            nGLCMr = np.ravel(nGLCM)
            x, y = np.mgrid[0:num_levels, 0:num_levels]
            xy = x*y
            ldata.at[r, 'Haralick.Correlation'] = \
                (np.dot(np.ravel(xy), nGLCMr) - np.square(meanx)) / variance

            # f3: computes sum of squares : variance
            ldata.at[r, 'Haralick.SumofSquar'] = variance

            # f4: computes inverse difference moment
            xy_IDM = 1. / (1+np.square(x-y))
            ldata.at[r, 'Haralick.IDM'] = \
                np.dot(np.ravel(xy_IDM), nGLCMr)

            # f5: computes sum average
            n_Plus = np.arange(2*num_levels-1)
            ldata.at[r, 'Haralick.SumAverage'] = \
                np.dot(n_Plus, pxPlusy)

            # f6: computes sum variance
            # [1] uses sum entropy, but we use sum average
            ldata.at[r, 'Haralick.SumVariance'] = \
                np.dot(np.square(n_Plus), pxPlusy) - \
                np.square(ldata.at[r, 'Haralick.SumAverage'])

            # f7: computes sum entropy
            ldata.at[r, 'Haralick.SumEntropy'] = \
                -np.dot(pxPlusy, np.log2(pxPlusy+e))

            # f8: computes entropy
            ldata.at[r, 'Haralick.Entropy'] = \
                -np.dot(nGLCMr, np.log2(nGLCMr+e))

            # f9: computes variance px-y
            ldata.at[r, 'Haralick.Variance'] = np.var(pxMinusy)

            # f10: computes difference entropy px-y
            ldata.at[r, 'Haralick.DifferenceEntropy'] = \
                -np.dot(pxMinusy, np.log2(pxMinusy+e))

            # f11: computes information measures of correlation
            # gets entropies of px and py
            HX = -np.dot(px, np.log2(px+e))
            HY = -np.dot(py, np.log2(py+e))
            HXY = ldata.at[r, 'Haralick.Entropy']
            pxy_ij = np.outer(px, py)
            pxy_ijr = np.ravel(pxy_ij)
            HXY1 = -np.dot(nGLCMr, np.log2(pxy_ijr+e))
            HXY2 = -np.dot(pxy_ijr, np.log2(pxy_ijr+e))
            ldata.at[r, 'Haralick.IMC1'] = (HXY-HXY1)/max(HX, HY)

            # f12: computes information measures of correlation
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
