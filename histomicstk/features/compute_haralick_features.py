import numpy as np

from ._compute_marginal_glcm_probs_cython import \
    _compute_marginal_glcm_probs_cython
from .graycomatrixext import (_default_num_levels, _default_offsets,
                              graycomatrixext)


def compute_haralick_features(im_label, im_intensity, offsets=None,
                              num_levels=None, gray_limits=None, rprops=None):
    r"""
    Calculates 26 Haralick texture features for each object in the given label
    mask.

    These features are derived from gray-level co-occurence matrix (GLCM)
    that is a two dimensional histogram containing the counts/probabilities of
    co-occurring intensity values with a given neighborhood offset in the
    region occupied by an object in the image.

    Parameters
    ----------
    im_label : array_like
        An ND labeled mask image wherein intensity of a pixel is the ID of the
        object it belongs to. Non-zero values are considered to be foreground
        objects.

    im_intensity : array_like
        An ND single channel intensity image.

    offsets : array_like, optional
        A (num_offsets, num_image_dims) array of offset vectors
        specifying the distance between the pixel-of-interest and
        its neighbor. Note that the first dimension corresponds to
        the rows.

        See `histomicstk.features.graycomatrixext` for more details.

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
    This function computes the following list of haralick features derived
    from normalized GLCMs (P) of the given list of neighborhood offsets:

    Haralick.ASM.Mean, Haralick.ASM.Range : float
        Mean and range of the angular second moment (ASM) feature for GLCMs
        of all offsets. It is a measure of image homogeneity and is computed
        as follows:

        .. math::

            ASM = \sum_{i,j=0}^{levels-1} p(i,j)^2

    Haralick.Contrast.Mean, Haralick.Contrast.Range : float
        Mean and range of the Contrast feature for GLCMs of all offsets. It is
        a measure of the amount of variation between intensities of
        neighboiring pixels. It is equal to zero for a constant image and
        increases as the amount of variation increases. It is computed as
        follows:

        .. math::

            Contrast = \sum_{i,j=0}^{levels-1}  (i-j)^2 p(i,j)

    Haralick.Correlation.Mean, Haralick.Correlation.Range : float
        Mean and range of the Correlation feature for GLCMs of all offsets. It
        is a measure of correlation between the intensity values of
        neighboring pixels. It is computed as follows:

        .. math::

            Correlation =
            \sum_{i,j=0}^{levels-1} p(i,j)\left[\frac{(i-\mu_i)
            (j-\mu_j)}{\sigma_i \sigma_j}\right]

    Haralick.SumOfSquares.Mean, Haralick.SumOfSquares.Range : float
        Mean and range of the SumOfSquares feature for GLCMs of all offsets.
        It is a measure of variance and is computed as follows:

        .. math::

            SumofSquare =
            \sum_{i,j=0}^{levels-1} (i - \mu)^2 p(i,j)

    Haralick.IDM.Mean, Haralick.IDM.Range : float
        Mean and range of the inverse difference moment (IDM) feature for
        GLCMS of all offsets. It is a measure of homogeneity and is computed
        as follows:

        .. math::
            IDM = \sum_{i,j=0}^{levels-1} \frac{1}{1 + (i - j)^2} p(i,j)

    Haralick.SumAverage.Mean, Haralick.SumAverage.Range : float
        Mean and range of sum average feature for GLCMs of all offsets.
        It is computed as follows:

        .. math::

            SumAverage = \sum_{k=2}^{2 levels} k p_{x+y}(k), \qquad where \\

            p_{x+y}(k) =
            \sum_{i,j=0}^{levels-1} \delta_{i+j, k} p(i,j) \\

            \delta_{m,n} = \left\{
            \begin{array}{11}
                1 & {\rm when ~} m=n \\
                0 & {\rm when ~} m \ne n
            \end{array}
            \right.

    Haralick.SumVariance.Mean, Haralick.SumVariance.Range : float
        Mean and range of sum variance feature for the GLCMS of all offsets.
        It is computed as follows:

        .. math::

            SumVariance =
            \sum_{k=2}^{2 levels} (k - SumEntropy) p_{x+y}(k)

    Haralick.SumEntropy.Mean, Haralick.SumEntropy.Range : float
        Mean and range of the sum entropy features for GLCMS of all offsets.
        It is computed as follows:

        .. math::

            SumEntropy =
            - \sum_{k=2}^{2 levels} p_{x+y}(k) \log(p_{x+y}(k))

    Haralick.Entropy.Mean, Haralick.Entropy.Range : float
        Mean and range of the entropy features for GLCMs of all offsets.
        It is computed as follows:

        .. math::

            Entropy = - \sum_{i,j=0}^{levels-1} p(i,j) \log(p(i,j))


    Haralick.DifferenceVariance.Mean, Haralick.DifferenceVariance.Range : float
        Mean and Range of the difference variance feature of GLCMs of all
        offsets. It is computed as follows:

        .. math::

            DifferenceVariance = {\rm variance \ of ~} p_{x-y}, \qquad where \\

            p_{x-y}(k) =
            \sum_{i,j=0}^{levels-1} \delta_{|i-j|, k} p(i,j)

    Haralick.DifferenceEntropy.Mean, Haralick.DifferenceEntropy.Range : float
        Mean and range of the difference entropy feature for GLCMS of all
        offsets. It is computed as follows:

        .. math::

            DifferenceEntropy = {\rm entropy \ of ~} p_{x-y}

    Haralick.IMC1.Mean, Haralick.IMC1.Range : float
        Mean and range of the first information measure of correlation
        feature for GLCMs of all offsets. It is computed as follows:

        .. math::

            IMC1 = \frac{HXY - HXY1}{\max(HX,HY)}, \qquad where \\

            HXY = -\sum_{i,j=0}^{levels-1} p(i,j) \log(p(i,j)) \\

            HXY1 = -\sum_{i,j=0}^{levels-1} p(i,j) \log(p_x(i) p_y(j)) \\

            HX = -\sum_{i=0}^{levels-1} p_x(i) \log(p_x(i))    \\

            HY = -\sum_{j=0}^{levels-1} p_y(j) \log(p_y(j))    \\

            p_x(i) = \sum_{j=1}^{levels} p(i,j) \\

            p_y(j) = \sum_{j=1}^{levels} p(i,j)

    Haralick.IMC2.Mean, Haralick.IMC2.Range : float
        Mean and range of the second information measure of correlation
        feature for GLCMs of all offsets. It is computed as follows:

        .. math::

            IMC2 = [1 - \exp(-2(HXY2 - HXY))]^{1/2}, \qquad where \\

            HXY2 = -\sum_{i,j=0}^{levels-1} p_x(i) p_y(j) \log(p_x(i) p_y(j))

    References
    ----------
    .. [#] Haralick, et al. "Textural features for image classification,"
       IEEE Transactions on Systems, Man, and Cybernatics, vol. 6,
       pp: 610-621, 1973.
    .. [#] Luis Pedro Coelho. "Mahotas: Open source software for scriptable
       computer vision," Journal of Open Research Software, vol 1, 2013.

    """
    import pandas as pd
    from skimage.measure import regionprops

    # List of feature names
    feature_list = [
        'Haralick.ASM',
        'Haralick.Contrast',
        'Haralick.Correlation',
        'Haralick.SumOfSquares',
        'Haralick.IDM',
        'Haralick.SumAverage',
        'Haralick.SumVariance',
        'Haralick.SumEntropy',
        'Haralick.Entropy',
        'Haralick.DifferenceVariance',
        'Haralick.DifferenceEntropy',
        'Haralick.IMC1',
        'Haralick.IMC2',
    ]

    agg_feature_list = []
    for fname in feature_list:
        agg_feature_list.append(fname + '.Mean')
        agg_feature_list.append(fname + '.Range')

    # num_levels
    if num_levels is None:
        num_levels = _default_num_levels(im_intensity)

    # check for consistent shapes between 'I' and 'Label'
    if im_intensity.shape != im_label.shape:
        err_str = 'Inputs I and Label must have same shape'
        raise ValueError(err_str)

    num_dims = len(im_intensity.shape)

    # offsets
    if offsets is None:
        # set default offset value
        offsets = _default_offsets(im_intensity)

    else:
        # check sanity
        if offsets.shape[1] != num_dims:
            err_str = 'Dimension mismatch between input image and offsets'
            raise ValueError(err_str)

    num_offsets = offsets.shape[0]

    # compute object properties if not provided
    if rprops is None:
        rprops = regionprops(im_label)

    # create pandas data frame containing the features for each object
    numLabels = len(rprops)
    fdata = pd.DataFrame(
        np.zeros((numLabels, len(agg_feature_list))), columns=agg_feature_list,
    )

    n_Minus = np.arange(num_levels)
    n_Plus = np.arange(2 * num_levels - 1)

    x, y = np.mgrid[0:num_levels, 0:num_levels]
    xy = x * y
    xy_IDM = 1.0 / (1 + np.square(x - y))

    e = 0.00001  # small positive constant to avoid log 0

    num_features = len(feature_list)

    # Initialize the array for aggregated features
    aggregated_features = np.zeros(
        (numLabels, 2 * num_features),
    )  # Alternating mean and range

    for i in range(numLabels):
        if rprops[i] is None:
            continue

        # get bounds of an intensity image
        minr, minc, maxr, maxc = rprops[i].bbox

        # grab nucleus mask
        subImage = im_intensity[minr: maxr + 1, minc: maxc + 1].astype(np.uint8)

        # gets GLCM or gray-tone spatial dependence matrix
        arrayGLCM = graycomatrixext(
            subImage,
            offsets=offsets,
            num_levels=num_levels,
            gray_limits=gray_limits,
            symmetric=True,
            normed=True,
        )

        features_per_offset = np.zeros((num_offsets, num_features))

        for r in range(num_offsets):
            nGLCM = arrayGLCM[:, :, r]

            # get marginal-probabilities
            px, py, pxPlusy, pxMinusy = _compute_marginal_glcm_probs_cython(nGLCM)

            # computes angular second moment
            ASM = np.sum(np.square(nGLCM))

            # computes contrast
            Contrast = np.dot(np.square(n_Minus), pxMinusy)

            # computes correlation
            # gets weighted mean and standard deviation of px and py
            meanx = np.dot(n_Minus, px)
            variance = np.dot(px, np.square(n_Minus)) - np.square(meanx)
            nGLCMr = np.ravel(nGLCM)
            Correlation = (np.dot(np.ravel(xy), nGLCMr) - np.square(meanx)) / variance

            # computes sum of squares : variance
            SumOfSquares = variance

            # computes inverse difference moment
            IDM = np.dot(np.ravel(xy_IDM), nGLCMr)

            # computes sum average
            SumAverage = np.dot(n_Plus, pxPlusy)

            # computes sum variance
            # [1] uses sum entropy, but we use sum average
            SumVariance = np.dot(np.square(n_Plus), pxPlusy) - np.square(SumAverage)

            # computes sum entropy
            SumEntropy = -np.dot(pxPlusy, np.log2(pxPlusy + e))

            # computes entropy
            Entropy = -np.dot(nGLCMr, np.log2(nGLCMr + e))

            # computes variance px-y
            DifferenceVariance = np.var(pxMinusy)

            # computes difference entropy px-y
            DifferenceEntropy = -np.dot(pxMinusy, np.log2(pxMinusy + e))

            # computes information measures of correlation
            # gets entropies of px and py
            HX = -np.dot(px, np.log2(px + e))
            HY = -np.dot(py, np.log2(py + e))
            HXY = Entropy
            pxy_ij = np.outer(px, py)
            pxy_ijr = np.ravel(pxy_ij)
            HXY1 = -np.dot(nGLCMr, np.log2(pxy_ijr + e))
            HXY2 = -np.dot(pxy_ijr, np.log2(pxy_ijr + e))
            IMC1 = (HXY - HXY1) / max(HX, HY)

            # computes information measures of correlation
            IMC2 = np.sqrt(np.maximum(0, 1 - np.exp(-2.0 * (HXY2 - HXY))))

            features_per_offset[r] = [
                ASM,
                Contrast,
                Correlation,
                SumOfSquares,
                IDM,
                SumAverage,
                SumVariance,
                SumEntropy,
                Entropy,
                DifferenceVariance,
                DifferenceEntropy,
                IMC1,
                IMC2,
            ]

            # Calculate means and ranges across all features in a vectorized manner
            means = np.mean(features_per_offset, axis=0)
            ranges = np.ptp(features_per_offset, axis=0)

            # Assign means and ranges to the aggregated_features array in alternating columns
            aggregated_features[i, ::2] = means
            aggregated_features[i, 1::2] = ranges

    # Preparing DataFrame columns with alternating mean and range suffixes
    fdata = pd.DataFrame(aggregated_features, columns=agg_feature_list)

    return fdata
