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

from pandas import DataFrame, concat
from skimage.color import rgb2gray
from skimage.segmentation import slic
from skimage.transform import resize

from histomicstk.saliency.tissue_detection import (
    get_slide_thumbnail, get_tissue_mask)
from histomicstk.annotations_and_masks.masks_to_annotations_handler import (
    get_contours_from_mask, get_annotation_documents_from_contours)
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    get_image_from_htk_response)

# %%===========================================================================

# get tissue mask
thumbnail_rgb = get_slide_thumbnail(gc, SAMPLE_SLIDE_ID)
labeled, _ = get_tissue_mask(
    thumbnail_rgb, deconvolve_first=False,
    n_thresholding_steps=1, sigma=1.5, min_size=500)

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
        'opacity': 0,
        'lineWidth': 4.0,
    }
    annotation_docs = get_annotation_documents_from_contours(
        contours_tissue.copy(), docnamePrefix='test', annprops=annprops,
        verbose=False, monitorPrefix="tissue : annotation docs")

    # deleting existing annotations in target slide (if any)
    existing_annotations = gc.get('/annotation/item/' + SAMPLE_SLIDE_ID)
    for ann in existing_annotations:
        gc.delete('/annotation/%s' % ann['_id'])

    # post annotations to slide
    _ = gc.post(
        "/annotation?itemId=" + SAMPLE_SLIDE_ID, json=annotation_docs[0])


_plot_and_post_tissue()

# %%===========================================================================

MAG = 3.0
slide_id = SAMPLE_SLIDE_ID

unique_tvals = list(set(np.unique(labeled)) - {0, })

tval = unique_tvals[0]
# for tval in unique_tvals:

# find coordinates at scan magnification
tloc = np.argwhere(labeled == tval)
ymin, xmin = [int(j) for j in np.min(tloc, axis=0) * F_tissue]
ymax, xmax = [int(j) for j in np.max(tloc, axis=0) * F_tissue]

# load RGB for this tissue piece at saliency magnification
getStr = "/item/%s/tiles/region?left=%d&right=%d&top=%d&bottom=%d" % (
    slide_id, xmin, xmax, ymin, ymax) + "&magnification=%d" % MAG
resp = gc.get(getStr, jsonResp=False)
tissue = get_image_from_htk_response(resp)
del resp

#tissue = rgb2gray(tissue)

# %%===========================================================================

# get superpixl mask
spixel_mask = slic(rgb2gray(tissue), n_segments=500, compactness=0.1)

# restrict to tissue mask
tmask = 0 + (labeled == tval)
tmask = tmask[
    int(ymin / F_tissue):int(ymax / F_tissue),
    int(xmin / F_tissue):int(xmax / F_tissue)]
tmask = resize(
    tmask, output_shape=spixel_mask.shape, order=0, preserve_range=True)
spixel_mask[tmask == 0] = 0

# Define GTCodes dataframe
GTCodes_df = DataFrame(columns=['group', 'GT_code', 'color'])

contours_df = DataFrame()

for spval in set(np.unique(spixel_mask)) - {0, }:
    spstr = 'spixel-%d' % spval
    GTCodes_df.loc[spstr, 'group'] = spstr
    GTCodes_df.loc[spstr, 'GT_code'] = spval
    GTCodes_df.loc[spstr, 'color'] = 'rgb(10,10,10)'

contours_df = get_contours_from_mask(
    MASK=spixel_mask, GTCodes_df=GTCodes_df,
    get_roi_contour=False, MIN_SIZE=0, MAX_SIZE=None, verbose=False,
    monitorPrefix=spstr)
contours_df.loc[:, "group"] = "spixel"


# get annotation docs
annprops = {
    'F': slide_info['magnification'] / MAG,
    'X_OFFSET': xmin,
    'Y_OFFSET': ymin,
    'opacity': 0,
    'lineWidth': 0,
}
annotation_docs = get_annotation_documents_from_contours(
    contours_df.copy(), docnamePrefix='test', annprops=annprops,
    annots_per_doc=1000, separate_docs_by_group=False,
    verbose=False, monitorPrefix="spixels : annotation docs")


_ = gc.post(
        "/annotation?itemId=" + SAMPLE_SLIDE_ID, json=annotation_docs[0])


# %% ==========================================================================










