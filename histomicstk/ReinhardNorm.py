import numpy as np
import RudermanLABFwd as rlf
import RudermanLABInv as rli


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
    RudermanLABFwd, RudermanLABInv

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
    I_LAB = rlf.RudermanLABFwd(I)

    # calculate SourceMu if not provided
    if SourceMu is None:
        SourceMu = I_LAB.sum(axis=0).sum(axis=0) / (m*n)
        print(SourceMu)

    # center to zero-mean
    I_LAB[:, :, 0] = I_LAB[:, :, 0] - SourceMu[0]
    I_LAB[:, :, 1] = I_LAB[:, :, 1] - SourceMu[1]
    I_LAB[:, :, 2] = I_LAB[:, :, 2] - SourceMu[2]

    # calculate SourceSigma if not provided
    if SourceSigma is None:
        SourceSigma = ((I_LAB*I_LAB).sum(axis=0).sum(axis=0) / (m*n-1)) ** 0.5
        print(SourceSigma)

    # scale to unit variance
    I_LAB[:, :, 0] = I_LAB[:, :, 0] / SourceSigma[0]
    I_LAB[:, :, 1] = I_LAB[:, :, 1] / SourceSigma[1]
    I_LAB[:, :, 2] = I_LAB[:, :, 2] / SourceSigma[2]

    # rescale and recenter to match target statistics
    I_LAB[:, :, 0] = I_LAB[:, :, 0] * TargetSigma[0] + TargetMu[0]
    I_LAB[:, :, 1] = I_LAB[:, :, 1] * TargetSigma[1] + TargetMu[1]
    I_LAB[:, :, 2] = I_LAB[:, :, 2] * TargetSigma[2] + TargetMu[2]

    # convert back to RGB colorspace
    I_Normalized = rli.RudermanLABInv(I_LAB)
    I_Normalized[I_Normalized > 255] = 255
    I_Normalized[I_Normalized < 0] = 0
    I_Normalized = I_Normalized.astype(np.uint8)
    
    return(I_Normalized)
