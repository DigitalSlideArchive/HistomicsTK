#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 18:09:17 2019.

@author: mtageld
"""
import numpy as np
from histomicstk.utils import (
    convert_image_to_matrix, convert_matrix_to_image)
from histomicstk.preprocessing.color_conversion import (
    rgb_to_sda, sda_to_rgb)


def augment_stain_concentration(
        StainsFloat, W, sda_fwd=None, I_0=None, mask_out=None,
        sigma1=0.8, sigma2=0.8):
    """

    References
    ----------
    .. [#] Tellez, David, Maschenka Balkenhol, Irene Otte-Höller,
           Rob van de Loo, Rob Vogels, Peter Bult, Carla Wauters et al.
           "Whole-slide mitosis detection in H&E breast histology using PHH3
           as a reference to train distilled stain-invariant convolutional
           networks." IEEE transactions on medical imaging 37, no. 9
           (2018): 2126-2136.
    .. [#] Tellez, David, Geert Litjens, Peter Bandi, Wouter Bulten,
           John-Melle Bokhorst, Francesco Ciompi, and Jeroen van der Laak.
           "Quantifying the effects of data augmentation and stain color
           normalization in convolutional neural networks for computational
           pathology." arXiv preprint arXiv:1902.06543 (2019).
    .. [#] Implementation inspired by Peter Byfield StainTools repository. See
           https://github.com/Peter554/StainTools/blob/master/LICENSE.txt
           for copyright license (MIT license).
    """
    # augment everything, otherwise only augment specific pixels
    if mask_out is None:
        keep_mask = np.zeros(StainsFloat.shape[:2]) == 0
    else:
        keep_mask = np.equal(mask_out, False)
    keep_mask = np.tile(keep_mask[..., None], (1, 1, 3))
    keep_mask = convert_image_to_matrix(keep_mask)

    # transform 3D input stain image to 2D stain matrix format
    m = convert_image_to_matrix(StainsFloat)

    # transform input stains to optical density values
    if sda_fwd is None:
        sda_fwd = rgb_to_sda(
            m, 255 if I_0 is not None else None, allow_negatives=True)

    # perturb concentrations in SDA space
    augmented_sda = sda_fwd.copy()
    for i in range(3):
        alpha = np.random.uniform(1 - sigma1, 1 + sigma1)
        beta = np.random.uniform(-sigma2, sigma2)
        augmented_sda[i, keep_mask[i, :]] *= alpha
        augmented_sda[i, keep_mask[i, :]] += beta

    # convolve with stains column vectors and convert to RGB
    sda_conv = np.dot(W, augmented_sda)
    sda_inv = sda_to_rgb(sda_conv, I_0)

    # reshape output, transform type
    augmented_rgb = (
        convert_matrix_to_image(sda_inv, StainsFloat.shape)
        .clip(0, 255).astype(np.uint8))

    return augmented_rgb


def rgb_augment_stain_concentration(im_rgb, I_0, *args, **kwargs):
    """"""
    im_sda = rgb_to_sda(im_rgb, I_0)
    return separate_stains_macenko_pca(im_sda, *args, **kwargs)

