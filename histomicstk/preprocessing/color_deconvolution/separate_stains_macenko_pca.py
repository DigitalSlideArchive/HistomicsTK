"""Placeholder."""
import numpy

import histomicstk.utils as utils

from . import _linalg as linalg
from .complement_stain_matrix import complement_stain_matrix


def separate_stains_macenko_pca(
        im_sda, minimum_magnitude=16, min_angle_percentile=0.01,
        max_angle_percentile=0.99, mask_out=None):
    """Compute the stain matrix for color deconvolution with the Macenko method.

    For a two-stain image or matrix in SDA space, this method works by
    computing a best-fit plane with PCA, wherein it selects the stain
    vectors as percentiles in the "angle distribution" in that plane.

    Parameters
    ----------
    im_sda : array_like
        Image (MxNx3) or matrix (3xN) in SDA space for which to compute the
        stain matrix.
    minimum_magnitude : float
        The magnitude below which vectors will be excluded from the computation
        of the angle distribution.

        The default is based on the paper value of 0.15, adjusted for our
        method of calculating SDA, thus 0.15 * 255 * log(10)/log(255)
    min_angle_percentile : float
        The smaller percentile of one of the vectors to pick from the angle
        distribution
    max_angle_percentile : float
        The larger percentile of one of the vectors to pick from the angle
        distribution
    mask_out : array_like
        if not None, should be (m, n) boolean numpy array.
        This parameter ensures exclusion of non-masked areas from calculations.
        This is relevant because elements like blood, sharpie marker,
        white space, etc may throw off the normalization somewhat.

    Returns
    -------
    w : array_like
        A 3x3 matrix of stain column vectors

    Note
    ----
    All input pixels not otherwise excluded are used in the computation of the
    principal plane and the angle distribution.

    See Also
    --------
    histomicstk.preprocessing.color_deconvolution.color_deconvolution
    histomicstk.preprocessing.color_deconvolution.separate_stains_xu_snmf

    References
    ----------
    .. [#] Van Eycke, Y. R., Allard, J., Salmon, I., Debeir, O., &
           Decaestecker, C. (2017).  Image processing in digital pathology: an
           opportunity to solve inter-batch variability of immunohistochemical
           staining.  Scientific Reports, 7.
    .. [#] Macenko, M., Niethammer, M., Marron, J. S., Borland, D.,
           Woosley, J. T., Guan, X., ... & Thomas, N. E. (2009, June).
           A method for normalizing histology slides for quantitative analysis.
           In Biomedical Imaging: From Nano to Macro, 2009.  ISBI'09.
           IEEE International Symposium on (pp. 1107-1110). IEEE.

    """
    # Image matrix
    m = utils.convert_image_to_matrix(im_sda)

    # mask out irrelevant values
    if mask_out is not None:
        keep_mask = numpy.equal(mask_out[..., None], False)
        keep_mask = numpy.tile(keep_mask, (1, 1, 3))
        keep_mask = utils.convert_image_to_matrix(keep_mask)
        m = m[:, keep_mask.all(axis=0)]

    # get rid of NANs and infinities
    m = utils.exclude_nonfinite(m)

    # Principal components matrix
    pcs = linalg.get_principal_components(m)
    # Input pixels projected into the PCA plane
    proj = pcs.T[:-1].dot(m)
    # Pixels above the magnitude threshold
    filt = proj[:, linalg.magnitude(proj) > minimum_magnitude]
    # The "angles"
    angles = _get_angles(filt)

    # The stain vectors

    def get_percentile_vector(p):
        return pcs[:, :-1].dot(filt[:, argpercentile(angles, p)])

    min_v = get_percentile_vector(min_angle_percentile)
    max_v = get_percentile_vector(max_angle_percentile)

    # The stain matrix
    w = complement_stain_matrix(linalg.normalize(
        numpy.array([min_v, max_v]).T))
    return w


def _get_angles(m):
    """Take a 2xN matrix of vectors and return a length-N array of an.

    ... angle-like quantity.
    Since this is an internal function, we assume that the values
    result from PCA, and so the second element of the vectors captures
    secondary variation -- and thus is the one that takes on both
    positive and negative values.

    """
    m = linalg.normalize(m)
    # "Angle" towards +x from the +y axis
    return (1 - m[1]) * numpy.sign(m[0])


def argpercentile(arr, p):
    """Calculate index in arr of element nearest the pth percentile."""
    # Index corresponding to percentile
    i = int(p * arr.size + 0.5)
    return numpy.argpartition(arr, i)[i]
