#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 02:31:32 2019.

@author: mtageld
"""
import numpy as np
from histomicstk.preprocessing.color_deconvolution.stain_color_map import (
    stain_color_map)
from histomicstk.preprocessing.color_deconvolution.find_stain_index import (
    find_stain_index)
from histomicstk.preprocessing.color_deconvolution.color_deconvolution import (
    color_deconvolution)
from histomicstk.preprocessing.color_deconvolution.color_convolution import (
    color_convolution)
from histomicstk.preprocessing.color_deconvolution import (
    complement_stain_matrix)
from histomicstk.preprocessing.color_deconvolution.\
    rgb_separate_stains_macenko_pca import rgb_separate_stains_macenko_pca
from histomicstk.preprocessing.color_deconvolution.\
    rgb_separate_stains_xu_snmf import rgb_separate_stains_xu_snmf


def deconvolution_based_normalization(
        im_src, W_target=None, im_target=None,
        stains=['hematoxylin', 'eosin'], W_source=None,
        stain_deconvolution_method='macenko_pca',
        stain_deconvolution_params={}):
    """Perform color normalization using color deconvolution to transform the
    color characteristics of an image to a desired standard.

    After the image is deconvolved into its component stains (eg, H&E), it is
    convolved with a stain column vectors matrix from the target image from
    which the color characteristics need to be transferred.

    Parameters
    ------------
    im_src : array_like
        An RGB image (m x n x 3) to colro normalize

    W_target : np array, default is None
        A 3x3 matrix of target stain column vectors. If not provided,
        and im_target is also not provided, the default behavior is to use
        histomicstk.preprocessing.color_deconvolution.stain_color_map
        to provide an idealized target matrix.

    im_target : array_like, default is None
        An RGB image (m x n x 3) that has good color properties that ought to
        be transferred to im_src. If you provide this parameter, im_target
        will be used to extract W_target and the W_target parameter will
        be ignored.

    stains : list, optional
        List of stain names (order is important). Default is H&E. This is
        particularly relevant in macenco where the order of stains is not
        preserved during stain unmixing, so this method uses
        histomicstk.preprocessing.color_deconvolution.find_stain_index
        to reorder the stains matrix to the order provided by this parameter

    W_source : np array, default is None
        A 3x3 matrix of source stain column vectors. Only provide this
        parameter if you know the stains matrix in advance (unlikely) and would
        like to perform supervised deconvolution. If this is not provided,
        or the stain_deconvolution_method parameter is something other than
        'supervised', stain_deconvolution_method is used to estimate W_source.

    stain_deconvolution_method : str, default is 'macenko_pca'
        stain deconvolution method to use. It should be one of the following
        'supervised', 'macenko_pca', or 'xu_snmf'.

    stain_deconvolution_params : dict, default is an empty dict
        kwargs to pass as-is to the stain deconvolution method. If you'd like
        to mask out certain areas from the strain matrix estimation, you may
        pass this here as, for eg, the mask_out parameter for macenko.

    Returns
    --------
    array_like
        Color Normalized RGB image (m x n x 3)


    See Also
    --------
    histomicstk.preprocessing.color_deconvolution.color_deconvolution
    histomicstk.preprocessing.color_convolution.color_convolution
    histomicstk.preprocessing.color_deconvolution.separate_stains_macenko_pca
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
    .. [#] Xu, J., Xiang, L., Wang, G., Ganesan, S., Feldman, M., Shih, N. N.,
           ...& Madabhushi, A. (2015). Sparse Non-negative Matrix Factorization
           (SNMF) based color unmixing for breast histopathological image
           analysis.  Computerized Medical Imaging and Graphics, 46, 20-29.
    """
    stain_deconvolution_method = stain_deconvolution_method.lower()

    if stain_deconvolution_method == 'supervised':
        assert W_source is not None, \
            "W_source must be provided for supervised deconvolution."

    elif stain_deconvolution_method == 'macenko_pca':
        stain_deconvolution = rgb_separate_stains_macenko_pca

    elif stain_deconvolution_method == 'xu_snmf':
        stain_deconvolution = rgb_separate_stains_xu_snmf

    else:
        raise ValueError("Unknown/Unimplemented deconvolution method.")

    # get W_source
    if stain_deconvolution_method != 'supervised':
        W_source = stain_deconvolution(im_src, **stain_deconvolution_params)

    # Get W_target

    if all(j is None for j in [W_target, im_target]):
        # Normalize to 'ideal' stain matrix if none is provided
        W_target = np.array([stain_color_map[stains[0]], stains[1]]).T
        W_target = complement_stain_matrix(W_target)

    elif im_target is not None:
        # Get W_target from target image
        W_target = stain_deconvolution(im_target, **stain_deconvolution_params)

    # If Macenco method, reorder channels in W_target and W_source as desired.
    # This is actually a necessary step in macenko's method since we're
    # not guaranteed the order of the different stains.
    if stain_deconvolution_method == 'macenko_pca':

        def _get_channel_order(W):
            first = find_stain_index(stain_color_map[stains[0]], W)
            second = 1 - first
            # third "stain" is cross product of 1st 2 channels
            # calculated using complement_stain_matrix()
            third = 2
            return first, second, third

        def _ordered_stack(mat, order):
            return np.stack([mat[..., j] for j in order], -1)

        def _reorder_stains(W):
            return _ordered_stack(W, _get_channel_order(W))

        W_target = _reorder_stains(W_target)
        W_source = _reorder_stains(W_source)

    # find stains matrix from source image
    _, StainsFloat, _ = color_deconvolution(im_src, w=W_source, I_0=None)

    # Convolve source image SainsFloat with W_target
    im_src_normalized = color_convolution(StainsFloat, W_target)

    return im_src_normalized
