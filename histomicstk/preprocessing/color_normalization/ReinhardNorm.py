from histomicstk.preprocessing import color_conversion
import numpy as np


def ReinhardNorm(I, TargetMu, TargetSigma, SourceMu=None, SourceSigma=None):
    """Performs Reinhard color normalization to transform the color
    characteristics of an image to a desired standard.

    The standard is defined by the mean and standard deviations of the target
    image in LAB color space defined by Ruderman. The input image is converted
    to Ruderman's LAB space, the LAB channels are each centered and scaled to
    zero-mean unit variance, and then rescaled and shifted to match the target
    image statistics. If the LAB statistics for the input image are provided
    (`SourceMu` and `SourceSigma`) then these will be used for normalization,
    otherwise they will be derived from the input image `I`.

    Parameters
    ----------
    I : array_like
        An RGB image
    TargetMu : array_like
        A 3-element array containing the means of the target image channels
        in LAB color space.
    TargetSigma : array_like
        A 3-element array containing the standard deviations of the target
        image channels in LAB color space.
    SourceMu : array_like, optional
        A 3-element array containing the means of the source image channels in
        LAB color space. Used with ReinhardSample for uniform normalization of
        tiles from a slide.
    SourceSigma : array, optional
        A 3-element array containing the standard deviations of the source
        image channels in LAB color space. Used with ReinhardSample for
        uniform normalization of tiles tiles from a slide.

    Returns
    -------
    I_Normalized : array_like
        Color Normalized RGB image with corrected color characteristics.

    See Also
    --------
    histomicstk.preprocessing.color_conversion.rgb_to_lab,
    histomicstk.preprocessing.color_conversion.lab_to_rgb

    References
    ----------
    .. [1] E. Reinhard, M. Adhikhmin, B. Gooch, P. Shirley, "Color transfer
       between images," in IEEE Computer Graphics and Applications, vol.21,
       no.5,pp.34-41, 2001.
    .. [2] D. Ruderman, T. Cronin, and C. Chiao, "Statistics of cone responses
       to natural images: implications for visual coding," J. Opt. Soc. Am. A
       vol.15, pp.2036-2045, 1998.
    """

    # get input image dimensions
    m = I.shape[0]
    n = I.shape[1]

    # convert input image to LAB color space
    I_LAB = color_conversion.rgb_to_lab(I)

    # calculate SourceMu if not provided
    if SourceMu is None:
        SourceMu = I_LAB.sum(axis=0).sum(axis=0) / (m * n)

    # center to zero-mean
    for i in range(3):
        I_LAB[:, :, i] = I_LAB[:, :, i] - SourceMu[i]

    # calculate SourceSigma if not provided
    if SourceSigma is None:
        SourceSigma = ((I_LAB * I_LAB).sum(axis=0).sum(axis=0) /
                       (m * n - 1)) ** 0.5

    # scale to unit variance
    for i in range(3):
        I_LAB[:, :, i] = I_LAB[:, :, i] / SourceSigma[i]

    # rescale and recenter to match target statistics
    for i in range(3):
        I_LAB[:, :, i] = I_LAB[:, :, i] * TargetSigma[i] + TargetMu[i]

    # convert back to RGB colorspace
    I_Normalized = color_conversion.lab_to_rgb(I_LAB)
    I_Normalized[I_Normalized > 255] = 255
    I_Normalized[I_Normalized < 0] = 0
    I_Normalized = I_Normalized.astype(np.uint8)

    return I_Normalized
