#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 02:31:32 2019.

@author: mtageld
"""
import numpy as np
from histomicstk.preprocessing.color_deconvolution.stain_color_map import (
    stain_color_map)
from histomicstk.preprocessing.color_deconvolution.color_deconvolution import (
    stain_unmixing_routine, color_deconvolution_routine)
from histomicstk.preprocessing.color_deconvolution.color_convolution import (
    color_convolution)
from histomicstk.preprocessing.color_deconvolution import (
    complement_stain_matrix)


def deconvolution_based_normalization(
        im_src, W_source=None, W_target=None, im_target=None,
        stains=None, mask_out=None, stain_unmixing_routine_params=None):
    """Perform color normalization using color deconvolution to transform the.

    ... color characteristics of an image to a desired standard.
    After the image is deconvolved into its component stains (eg, H&E), it is
    convolved with a stain column vectors matrix from the target image from
    which the color characteristics need to be transferred.

    Parameters
    ------------
    im_src : array_like
        An RGB image (m x n x 3) to color normalize

    W_source : np array, default is None
        A 3x3 matrix of source stain column vectors. Only provide this
        if you know the stains matrix in advance (unlikely) and would
        like to perform supervised deconvolution. If this is not provided,
        stain_unmixing_routine() is used to estimate W_source.

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

    mask_out : array_like, default is None
        if not None, should be (m x n) boolean numpy array.
        This parameter ensures exclusion of non-masked areas from calculations
        and normalization. This is relevant because elements like blood,
        sharpie marker, white space, etc may throw off the normalization.

    stain_unmixing_routine_params : dict, default is empty dict
        k,v for stain_unmixing_routine().

    Returns
    --------
    array_like
        Color Normalized RGB image (m x n x 3)


    See Also
    --------
    histomicstk.preprocessing.color_deconvolution.color_deconvolution_routine
    histomicstk.preprocessing.color_convolution.color_convolution

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
    stains = ['hematoxylin', 'eosin'] if stains is None else stains
    stain_unmixing_routine_params = (
        {} if stain_unmixing_routine_params is None else
        stain_unmixing_routine_params)
    for k in ['W_source', 'mask_out']:
        assert k not in stain_unmixing_routine_params.keys(), \
            "%s must be provided as a separate parameter." % k

    # find stains matrix from source image
    stain_unmixing_routine_params['stains'] = stains
    _, StainsFloat, _ = color_deconvolution_routine(
        im_src, W_source=W_source, mask_out=mask_out,
        **stain_unmixing_routine_params)

    # Get W_target

    if all(j is None for j in [W_target, im_target]):
        # Normalize to 'ideal' stain matrix if none is provided
        W_target = np.array(
            [stain_color_map[stains[0]], stain_color_map[stains[1]]]).T
        W_target = complement_stain_matrix(W_target)

    elif im_target is not None:
        # Get W_target from target image
        W_target = stain_unmixing_routine(
            im_target, **stain_unmixing_routine_params)

    # Convolve source image StainsFloat with W_target
    im_src_normalized = color_convolution(StainsFloat, W_target)

    # return masked values using unnormalized image
    if mask_out is not None:
        keep_mask = np.not_equal(mask_out, True)
        for i in range(3):
            original = im_src[:, :, i].copy()
            new = im_src_normalized[:, :, i].copy()
            original[keep_mask] = 0
            new[mask_out] = 0
            im_src_normalized[:, :, i] = new + original

    return im_src_normalized
