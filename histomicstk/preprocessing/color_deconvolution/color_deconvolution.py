"""Placeholder."""
import collections

import numpy as np

import histomicstk.utils as utils
from histomicstk.preprocessing import color_conversion
from histomicstk.preprocessing.color_deconvolution.find_stain_index import \
    find_stain_index
from histomicstk.preprocessing.color_deconvolution.rgb_separate_stains_macenko_pca import \
    rgb_separate_stains_macenko_pca
from histomicstk.preprocessing.color_deconvolution.rgb_separate_stains_xu_snmf import \
    rgb_separate_stains_xu_snmf
from histomicstk.preprocessing.color_deconvolution.stain_color_map import \
    stain_color_map

from ._linalg import normalize
from .complement_stain_matrix import complement_stain_matrix


def color_deconvolution(im_rgb, w, I_0=None):
    """Perform color deconvolution.

    The given RGB Image `I` is first first transformed into optical density
    space, and then projected onto the stain vectors in the columns of the
    3x3 stain matrix `W`.

    For deconvolving H&E stained image use:

    `w` = array([[0.650, 0.072, 0], [0.704, 0.990, 0], [0.286, 0.105, 0]])

    Parameters
    ----------
    im_rgb : array_like
        Input RGB Image that needs to be deconvolved.
    w : array_like
        A 3x3 matrix containing the color vectors in columns.
        For two stain images the third column is zero and will be
        complemented using cross-product. Atleast two of the three
        columns must be non-zero.
    I_0 : float or array_like, optional
        A float a 3-vector containing background RGB intensities.
        If unspecified, use the old OD conversion.

    Returns
    -------
    Stains : array_like
        An rgb image where in each channel contains the image of the
        stain of the corresponding column in the stain matrix `W`.
        The intensity range of each channel is [0, 255] suitable for
        displaying.
    StainsFloat : array_like
        An intensity image of deconvolved stains that is unbounded,
        suitable for reconstructing color images of deconvolved stains
        with color_convolution.
    Wc : array_like
        A 3x3 complemented stain matrix. Useful for color image
        reconstruction with color_convolution.

    See Also
    --------
    histomicstk.preprocessing.color_deconvolution.complement_stain_matrix,
    histomicstk.preprocessing.color_deconvolution.color_convolution
    histomicstk.preprocessing.color_conversion.rgb_to_od
    histomicstk.preprocessing.color_conversion.od_to_rgb
    histomicstk.preprocessing.color_conversion.rgb_to_sda
    histomicstk.preprocessing.color_conversion.sda_to_rgb

    """
    # complement stain matrix if needed
    if np.linalg.norm(w[:, 2]) <= 1e-16:
        wc = complement_stain_matrix(w)
    else:
        wc = w

    # normalize stains to unit-norm
    wc = normalize(wc)

    # invert stain matrix
    Q = np.linalg.inv(wc)

    # transform 3D input image to 2D RGB matrix format
    m = utils.convert_image_to_matrix(im_rgb)[:3]

    # transform input RGB to optical density values and deconvolve,
    # tfm back to RGB
    sda_fwd = color_conversion.rgb_to_sda(m, I_0)
    sda_deconv = np.dot(Q, sda_fwd)
    sda_inv = color_conversion.sda_to_rgb(sda_deconv,
                                          255 if I_0 is not None else None)

    # reshape output
    StainsFloat = utils.convert_matrix_to_image(sda_inv, im_rgb.shape)

    # transform type
    Stains = StainsFloat.clip(0, 255).astype(np.uint8)

    # return
    Unmixed = collections.namedtuple('Unmixed',
                                     ['Stains', 'StainsFloat', 'Wc'])
    Output = Unmixed(Stains, StainsFloat, wc)

    return Output


def _reorder_stains(W, stains=None):
    """Reorder stains in a stain matrix to a specific order.

    This is particularly relevant in macenco where the order of stains is not
    preserved during stain unmixing, so this method uses
    histomicstk.preprocessing.color_deconvolution.find_stain_index
    to reorder the stains matrix to the order provided by this parameter

    Parameters
    ------------
    W : np array
        A 3x3 matrix of stain column vectors.
    stains : list, optional
        List of stain names (order is important). Default is H&E.

    Returns
    ------------
    np array
        A re-ordered 3x3 matrix of stain column vectors.

    """
    stains = ['hematoxylin', 'eosin'] if stains is None else stains

    assert len(stains) == 2, "Only two-stain matrices are supported for now."

    def _get_channel_order(W):
        first = find_stain_index(stain_color_map[stains[0]], W)
        second = 1 - first
        # If 2 stains, third "stain" is cross product of 1st 2 channels
        # calculated using complement_stain_matrix()
        third = 2
        return first, second, third

    def _ordered_stack(mat, order):
        return np.stack([mat[..., j] for j in order], -1)

    return _ordered_stack(W, _get_channel_order(W))


def stain_unmixing_routine(
        im_rgb, stains=None, stain_unmixing_method='macenko_pca',
        stain_unmixing_params=None, mask_out=None):
    """Perform stain unmixing using the method of choice (wrapper).

    Parameters
    ------------
    im_rgb : array_like
        An RGB image (m x n x 3) to unmix.

    stains : list, optional
        List of stain names (order is important). Default is H&E. This is
        particularly relevant in macenco where the order of stains is not
        preserved during stain unmixing, so this method uses
        histomicstk.preprocessing.color_deconvolution.find_stain_index
        to reorder the stains matrix to the order provided by this parameter

    stain_unmixing_method : str, default is 'macenko_pca'
        stain unmixing method to use. It should be one of the following
        'macenko_pca', or 'xu_snmf'.

    stain_unmixing_params : dict, default is an empty dict
        kwargs to pass as-is to the stain unmixing method.

    mask_out : array_like, default is None
        if not None, should be (m x n) boolean numpy array.
        This parameter ensures exclusion of non-masked areas from calculations
        and normalization. This is relevant because elements like blood,
        sharpie marker, white space, etc may throw off the normalization.

    Returns
    --------
    Wc : array_like
        A 3x3 complemented stain matrix.

    See Also
    --------
    histomicstk.preprocessing.color_deconvolution.separate_stains_macenko_pca
    histomicstk.preprocessing.color_deconvolution.separate_stains_xu_snmf

    References
    ----------
    .. [#] Macenko, M., Niethammer, M., Marron, J. S., Borland, D.,
           Woosley, J. T., Guan, X., ... & Thomas, N. E. (2009, June).
           A method for normalizing histology slides for quantitative analysis.
           In Biomedical Imaging: From Nano to Macro, 2009.  ISBI'09.
           IEEE International Symposium on (pp. 1107-1110). IEEE.
    .. [#] Xu, J., Xiang, L., Wang, G., Ganesan, S., Feldman, M., Shih, N. N.,
           ...& Madabhushi, A. (2015). Sparse Non-negative Matrix Factorization
           (SNMF) based color unmixing for breast histopathological image
           analysis.  Computerized Medical Imaging and Graphics, 46, 20-29.

    """
    stains = ['hematoxylin', 'eosin'] if stains is None else stains
    stain_unmixing_params = {} if stain_unmixing_params is None else stain_unmixing_params

    stain_unmixing_method = stain_unmixing_method.lower()

    if stain_unmixing_method == 'macenko_pca':
        stain_deconvolution = rgb_separate_stains_macenko_pca
        stain_unmixing_params['I_0'] = None
        stain_unmixing_params['mask_out'] = mask_out

    elif stain_unmixing_method == 'xu_snmf':
        stain_deconvolution = rgb_separate_stains_xu_snmf
        stain_unmixing_params['I_0'] = None
        assert mask_out is None, "Masking is not yet implemented in xu_snmf."

    else:
        raise ValueError("Unknown/Unimplemented deconvolution method.")

    # get W_source
    W_source = stain_deconvolution(im_rgb, **stain_unmixing_params)

    # If Macenco method, reorder channels in W_target and W_source as desired.
    # This is actually a necessary step in macenko's method since we're
    # not guaranteed the order of the different stains.
    if stain_unmixing_method == 'macenko_pca':
        W_source = _reorder_stains(W_source, stains=stains)

    return W_source


def color_deconvolution_routine(
        im_rgb, W_source=None, mask_out=None, **kwargs):
    """Unmix stains mixing followed by deconvolution (wrapper).

    Parameters
    ------------
    im_rgb : array_like
        An RGB image (m x n x 3) to color normalize

    W_source : np array, default is None
        A 3x3 matrix of source stain column vectors. Only provide this
        if you know the stains matrix in advance (unlikely) and would
        like to perform supervised deconvolution. If this is not provided,
        stain_unmixing_routine() is used to estimate W_source.

    mask_out : array_like, default is None
        if not None, should be (m x n) boolean numpy array.
        This parameter ensures exclusion of non-masked areas from calculations
        and stain matrix. This is relevant because elements like blood,
        sharpie marker, white space, cannot be modeled as a mix of two stains.

    kwargs : k,v pairs
        Passed as-is to stain_unmixing_routine() if W_source is None.

    Returns
    --------
    Output from color_deconvolution()

    See Also
    --------
    histomicstk.preprocessing.color_deconvolution.stain_unmixing_routine
    histomicstk.preprocessing.color_deconvolution.color_deconvolution

    """
    # get W_source if not provided
    if W_source is None:
        W_source = stain_unmixing_routine(im_rgb, mask_out=mask_out, **kwargs)

    # deconvolve
    Stains, StainsFloat, wc = color_deconvolution(im_rgb, w=W_source, I_0=None)

    # mask out (keep in mind, image is inverted)
    if mask_out is not None:
        for i in range(3):
            Stains[..., i][mask_out] = 255
            StainsFloat[..., i][mask_out] = 255.

    return Stains, StainsFloat, wc
