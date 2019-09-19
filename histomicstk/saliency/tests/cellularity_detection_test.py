#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 02:25:34 2019.

@author: mtageld
"""

import girder_client
import numpy as np

# %%===========================================================================
# Constants & prep work

APIURL = 'http://candygram.neurology.emory.edu:8080/api/v1/'
SAMPLE_SLIDE_ID = "5d586d76bd4404c6b1f286ae"

gc = girder_client.GirderClient(apiUrl=APIURL)
# gc.authenticate(interactive=True)
gc.authenticate(apiKey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')


# %%===========================================================================

import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap

from pandas import DataFrame

from histomicstk.saliency.tissue_detection import (
    get_slide_thumbnail, get_tissue_mask)
from histomicstk.annotations_and_masks.masks_to_annotations_handler import (
    get_contours_from_mask, get_annotation_documents_from_contours)

# %%===========================================================================

# get tissue mask
thumbnail_rgb = get_slide_thumbnail(gc, SAMPLE_SLIDE_ID)
labeled, _ = get_tissue_mask(
    thumbnail_rgb, deconvolve_first=True,
    n_thresholding_steps=3, sigma=1., min_size=800)

# Find size relative to WSI
slide_info = gc.get('item/%s/tiles' % SAMPLE_SLIDE_ID)
F_tissue = slide_info['sizeX'] / labeled.shape[1]

# %%===========================================================================


def _plot_and_post_tissue():

    # Define color map
    vals = np.random.rand(256, 3)
    vals[0, ...] = [0.9, 0.9, 0.9]
    cMap = ListedColormap(1 - vals)

    TISSUE_COLOR = 'rgb(0,0,0)'

    # visualize
    f, ax = plt.subplots(1, 2, figsize=(20, 20))
    ax[0].imshow(thumbnail_rgb)
    ax[1].imshow(labeled, cmap=cMap)
    plt.show()

    # Define GTCodes dataframe
    GTCodes_df = DataFrame(columns=['group', 'GT_code', 'color'])
    GTCodes_df.loc['tissue', 'group'] = 'tissue'
    GTCodes_df.loc['tissue', 'GT_code'] = 1
    GTCodes_df.loc['tissue', 'color'] = TISSUE_COLOR

    # get annotation docs
    contours_tissue = get_contours_from_mask(
        MASK=0 + (labeled > 0), GTCodes_df=GTCodes_df,
        get_roi_contour=False, MIN_SIZE=0, MAX_SIZE=None, verbose=False,
        monitorPrefix="tissue: getting contours")
    annprops = {
        'F': F_tissue,
        'X_OFFSET': 0,
        'Y_OFFSET': 0,
        'opacity': 0.2,
        'lineWidth': 4.0,
    }
    annotation_docs = get_annotation_documents_from_contours(
        contours_tissue.copy(), docnamePrefix='test', annprops=annprops,
        verbose=False, monitorPrefix="tissue : annotation docs")

    # deleting existing annotations in target slide (if any)
    existing_annotations = gc.get('/annotation/item/' + SAMPLE_SLIDE_ID)
    for ann in existing_annotations:
        gc.delete('/annotation/%s' % ann['_id'])

    # post annotations to slide -- make sure it posts without errors
    _ = gc.post(
        "/annotation?itemId=" + SAMPLE_SLIDE_ID, json=annotation_docs[0])


_plot_and_post_tissue()

# %%===========================================================================





