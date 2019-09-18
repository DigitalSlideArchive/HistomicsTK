#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 00:06:28 2019

@author: mtageld
"""

# import unittest

import os
import girder_client
# from pandas import read_csv
# from imageio import imread
# import tempfile
# import shutil

import numpy as np
from PIL import Image
from matplotlib import pylab as plt
from matplotlib.colors import ListedColormap
# from imageio import imwrite
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
#
# =============================================================================


def get_slide_thumbnail(gc, slide_id):
    """Get slide thumbnail using girder client."""
    getStr = "/item/%s/tiles/thumbnail" % (slide_id)
    resp = gc.get(getStr, jsonResp=False)
    return get_image_from_htk_response(resp)


def deconv_color(im, stain_matrix_method="PCA"):
    """Convenience wrapper around color_deconvolution for H&E.

    See tutorial at:  examples/color-deconvolution.html
    """
    # Constant -- see documentation for color_deconvolution method
    stain_color_map = {
        'hematoxylin': [0.65, 0.70, 0.29],
        'eosin':       [0.07, 0.99, 0.11],
        'dab':         [0.27, 0.57, 0.78],
        'null':        [0.0, 0.0, 0.0],
        'HE_null':     [0.286, 0.105, 0],
    }
    I_0 = 255
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
        getting rid of sharpie marker. Sometimes things work better
        without it, though.
    stain_matrix_method - see deconv_color method in seed_utils
        n_thresholding_steps - int
        sigma - int

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
        Stains, channel = deconv_color(
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

    # each connected component gets a unique value
    unique, counts = np.unique(labeled[labeled > 0], return_counts=True)
    mask = labeled == unique[np.argmax(counts)]

    return mask, labeled

# %%===========================================================================
# Constants & prep work
# =============================================================================


APIURL = 'http://candygram.neurology.emory.edu:8080/api/v1/'
# SAMPLE_SLIDE_ID = '5d586d57bd4404c6b1f28640'
SAMPLE_SLIDE_ID = "5d817f5abd4404c6b1f744bb"

gc = girder_client.GirderClient(apiUrl=APIURL)
# gc.authenticate(interactive=True)
gc.authenticate(apiKey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')


# %%===========================================================================
#
# =============================================================================

thumbnail_rgb = get_slide_thumbnail(gc, SAMPLE_SLIDE_ID)

#%%
mask, labeled = get_tissue_mask(
    thumbnail_rgb, deconvolve_first=True,
    n_thresholding_steps=2, sigma=0.)

#%%

#vals = np.random.rand(256,3)
#vals[0, ...] = [0., 0., 0.]
#cMap = ListedColormap(1 - vals)

f, ax = plt.subplots(1, 3, figsize=(20, 20))
ax[0].imshow(thumbnail_rgb)
ax[1].imshow(mask)
ax[2].imshow(labeled == 200)
plt.show()




















