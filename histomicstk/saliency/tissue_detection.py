#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 03:29:24 2019.

@author: mtageld
"""

import numpy as np
from PIL import Image
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    get_image_from_htk_response)
from histomicstk.preprocessing.color_deconvolution.color_deconvolution import (
    color_deconvolution)
from histomicstk.preprocessing.color_deconvolution.\
    rgb_separate_stains_macenko_pca import rgb_separate_stains_macenko_pca
from histomicstk.preprocessing.color_deconvolution.find_stain_index import (
    find_stain_index)
import cv2
from skimage.filters import threshold_otsu, gaussian
from scipy import ndimage

Image.MAX_IMAGE_PIXELS = None

# %%===========================================================================


def get_slide_thumbnail(gc, slide_id):
    """Get slide thumbnail using girder client.

    Parameters
    -------------
    gc : object
        girder client to use
    slide_id : str
        girder ID of slide

    Returns
    ---------
    np array
        RGB slide thumbnail at lowest level

    """
    getStr = "/item/%s/tiles/thumbnail" % (slide_id)
    resp = gc.get(getStr, jsonResp=False)
    return get_image_from_htk_response(resp)

# %%===========================================================================


def _deconv_color(im, stain_matrix_method="PCA"):
    """Deconvolve using wrapper around color_deconvolution for H&E.

    See tutorial at:  examples/color-deconvolution.html

    Parameters
    ------------
    im : np array
        rgb image
    stain_matrix_method : str
        Currently only PCA supported, but the original method supports others.

    """
    # Constant -- see documentation for color_deconvolution method
    stain_color_map = {
        'hematoxylin': [0.65, 0.70, 0.29],
        'eosin':       [0.07, 0.99, 0.11],
        'dab':         [0.27, 0.57, 0.78],
        'null':        [0.0, 0.0, 0.0],
        'HE_null':     [0.286, 0.105, 0],
    }
    I_0 = None
    if stain_matrix_method == "PCA":  # Visually shows best results
        W_est = rgb_separate_stains_macenko_pca(im, I_0)
        Stains, _, _ = color_deconvolution(im_rgb=im, w=W_est, I_0=I_0)

        # Unlike SNMF, we're not guaranteed the order of the different stains.
        # find_stain_index guesses which one we want
        channel = find_stain_index(stain_color_map['hematoxylin'], W_est)
    else:
        raise NotImplementedError(
            """Not yet implemented here, but you can easily implement it
            yourself if you follow this tutorial:
            examples/color-deconvolution.html""")

    return Stains, channel

# %%===========================================================================


def get_tissue_mask(
        thumbnail_rgb, deconvolve_first=False, stain_matrix_method="PCA",
        n_thresholding_steps=1, sigma=0., min_size=500):
    """Get binary tissue mask from slide thumbnail.

    Parameters
    -----------
    thumbnail_rgb : np array
        (m, n, 3) nd array of thumbnail RGB image
    deconvolve_first : bool
        use hematoxylin channel to find cellular areas?
        This will make things ever-so-slightly slower but is better in
        getting rid of sharpie marker (if it's green, for example).
        Sometimes things work better without it, though.
    stain_matrix_method : str
        see deconv_color method in seed_utils
    n_thresholding_steps : int
        number of gaussian smoothign steps
    sigma : float
        sigma of gaussian filter
    min_size : int
        minimum size (in pixels) of contiguous tissue regions to keep

    Returns
    --------
    np bool array
        largest contiguous tissue region.
    np int32 array
        each unique value represents a unique tissue region

    """
    if deconvolve_first:
        # deconvolvve to ge hematoxylin channel (cellular areas)
        # hematoxylin channel return shows MINIMA so we invert
        Stains, channel = _deconv_color(
            thumbnail_rgb, stain_matrix_method=stain_matrix_method)
        thumbnail = 255 - Stains[..., channel]
    else:
        # grayscale thumbnail (inverted)
        thumbnail = 255 - cv2.cvtColor(thumbnail_rgb, cv2.COLOR_BGR2GRAY)

    for _ in range(n_thresholding_steps):

        # gaussian smoothing of grayscale thumbnail
        if sigma > 0.0:
            thumbnail = gaussian(
                thumbnail, sigma=sigma,
                output=None, mode='nearest', preserve_range=True)

        # get threshold to keep analysis region
        try:
            thresh = threshold_otsu(thumbnail[thumbnail > 0])
        except ValueError:  # all values are zero
            thresh = 0

        # replace pixels outside analysis region with upper quantile pixels
        thumbnail[thumbnail < thresh] = 0

    # convert to binary
    mask = 0 + (thumbnail > 0)

    # find connected components
    labeled, _ = ndimage.label(mask)

    # only keep
    unique, counts = np.unique(labeled[labeled > 0], return_counts=True)
    discard = np.in1d(labeled, unique[counts < min_size])
    discard = discard.reshape(labeled.shape)
    labeled[discard] = 0

    # largest tissue region
    mask = labeled == unique[np.argmax(counts)]

    return labeled, mask


# %%===========================================================================
