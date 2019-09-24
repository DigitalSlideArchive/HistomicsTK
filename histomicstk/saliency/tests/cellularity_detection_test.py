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
from sklearn.mixture import GaussianMixture
from skimage.measure import regionprops
from matplotlib import cm

from histomicstk.utils.general_utils import Base_HTK_Class

from histomicstk.saliency.tissue_detection import (
    get_slide_thumbnail, get_tissue_mask, _deconv_color)
from histomicstk.annotations_and_masks.masks_to_annotations_handler import (
    get_contours_from_mask, get_annotation_documents_from_contours)
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    get_image_from_htk_response)
from histomicstk.features.compute_intensity_features import (
    compute_intensity_features)
from histomicstk.features.compute_haralick_features import (
    compute_haralick_features)

# %%===========================================================================


class Cellularity_Detector_Superpixels(Base_HTK_Class):
    """Detect cellular regions in a slides by classifying superpixels."""

    def __init__(self, gc, slide_id, **kwargs):
        """Init Polygon_merger object.

        Arguments:
        -----------
        gc : object
            girder client object
        slide_id : str
            girder ID of slide
        MAG : float
            magnification at which to detect cellularity
        verbose : int
            0 - Do not print to screen
            1 - Print only key messages
            2 - Print everything to screen
        monitorPrefix : str
            text to prepend to printed statements

        """
        default_attr = {
            'MAG': 3.0,
            'verbose': 1,
            'monitorPrefix': "",
            'spixel_size_baseMag': 350 * 350,
            'compactness': 0.1,
            'use_grayscale': True,
            'use_intensity': True,
            'use_texture': False,
            # 'keep_feats': None,
            'keep_feats': [
                "Intensity.Mean", "Intensity.Median",
                "Intensity.Std", "Intensity.IQR",
                "Intensity.HistEntropy",
            ],
            'opacity': 0,
            'lineWidth': 3.0,
        }
        super().__init__(default_attr=default_attr)



    # %% =====================================================================



# %%

cds = Cellularity_Detector_Superpixels(gc, SAMPLE_SLIDE_ID)

# %%

a

# %%===========================================================================
# %%===========================================================================
# %%===========================================================================

slide_id = SAMPLE_SLIDE_ID

# %%===========================================================================

# get tissue mask
thumbnail_rgb = get_slide_thumbnail(gc, SAMPLE_SLIDE_ID)
labeled, _ = get_tissue_mask(
    thumbnail_rgb, deconvolve_first=False,
    n_thresholding_steps=1, sigma=1.5, min_size=500)

# Find size relative to WSI
slide_info = gc.get('item/%s/tiles' % SAMPLE_SLIDE_ID)
F_tissue = slide_info['sizeX'] / labeled.shape[1]

unique_tvals = list(set(np.unique(labeled)) - {0, })

tval = unique_tvals[1]
#for tval in unique_tvals:

# %%===========================================================================

tissue_mask = labeled == tval

def _get_tissue_rgb_from_small_mask(tissue_mask):
    # find coordinates at scan magnification
    tloc = np.argwhere(tissue_mask)
    ymin, xmin = [int(j) for j in np.min(tloc, axis=0) * F_tissue]
    ymax, xmax = [int(j) for j in np.max(tloc, axis=0) * F_tissue]
    # load RGB for this tissue piece at saliency magnification
    getStr = "/item/%s/tiles/region?left=%d&right=%d&top=%d&bottom=%d" % (
        slide_id, xmin, xmax, ymin, ymax) + "&magnification=%d" % MAG
    resp = gc.get(getStr, jsonResp=False)
    tissue = get_image_from_htk_response(resp)
    return tissue


# %%===========================================================================



if use_grayscale:
    tissue = rgb2gray(tissue)

spixel_size = spixel_size_baseMag * (MAG / slide_info['magnification'])
n_spixels = int(tissue.shape[0] * tissue.shape[1] / spixel_size)

# get superpixl mask
spixel_mask = slic(tissue, n_segments=n_spixels, compactness=compactness)

# restrict to tissue mask
tmask = 0 + (labeled == tval)
tmask = tmask[
    int(ymin / F_tissue):int(ymax / F_tissue),
    int(xmin / F_tissue):int(xmax / F_tissue)]
tmask = resize(
    tmask, output_shape=spixel_mask.shape, order=0, preserve_range=True)
spixel_mask[tmask == 0] = 0

# %% ==========================================================================

# deconvolvve to ge hematoxylin channel (cellular areas)
# hematoxylin channel return shows MINIMA so we invert
Stains, channel = _deconv_color(tissue)
tissue_htx = 255 - Stains[..., channel]

# %% ==========================================================================

# sanity checks
assert (use_intensity or use_texture)

# calculate features from superpixels -- using hematoxylin channel
rprops = regionprops(spixel_mask)
fdata_list = []
if use_intensity:
    fdata_list.append(compute_intensity_features(
        im_label=spixel_mask, im_intensity=tissue_htx, rprops=rprops))
if use_texture:
    fdata_list.append(compute_haralick_features(
        im_label=spixel_mask, im_intensity=tissue_htx, rprops=rprops))
fdata = concat(fdata_list, axis=1)

if keep_feats is not None:
    fdata = fdata.loc[:, keep_feats]

# Index is corresponding pixel value in the superpixel mask
# IMPORTANT -- this assumes that regionprops output is sorted by the unique
# pixel values in label mask, which it is by default
fdata.index = set(np.unique(spixel_mask)) - {0, }

# %% ==========================================================================

n_gaussian_components = 5

# Fit a 2? 3? 4? component gaussian mixture model to features
mmodel = GaussianMixture(n_components=n_gaussian_components)
spixel_labels = mmodel.fit_predict(fdata.values) + 1

# %% ==========================================================================

normalize_colors = True
cMap = cm.seismic

# Rank clusters by cellularity (intensity of hematocylin channel)

cluster_props = dict()

fdata.loc[:, "cluster"] = spixel_labels
for clid in np.unique(spixel_labels):
    cluster_props[clid] = {
        'cellularity': int(np.median(
            fdata.loc[fdata.loc[:, "cluster"] == clid, "Intensity.Median"]
            ) / 255 * 100),
    }

# assign colors
max_cellularity = max([j['cellularity'] for _, j in cluster_props.items()])
for clid in np.unique(spixel_labels):
    rgb = cMap(int(
            cluster_props[clid]['cellularity'] / max_cellularity * 255))[:-1]
    rgb = [int(255 * j) for j in rgb]
    cluster_props[clid]['color'] = 'rgb(%d,%d,%d)' % tuple(rgb)

# %% ==========================================================================

# Visualize single superpixels by cellularity

# Define GTCodes dataframe
GTCodes_df = DataFrame(columns=['group', 'GT_code', 'color'])
for spval, sp in fdata.iterrows():
    spstr = 'spixel-%d_cellularity-%d' % (
        spval, cluster_props[sp['cluster']]['cellularity'])
    GTCodes_df.loc[spstr, 'group'] = spstr
    GTCodes_df.loc[spstr, 'GT_code'] = spval
    GTCodes_df.loc[spstr, 'color'] = cluster_props[sp['cluster']]['color']

# get contours df
contours_df = get_contours_from_mask(
    MASK=spixel_mask, GTCodes_df=GTCodes_df,
    get_roi_contour=False, MIN_SIZE=0, MAX_SIZE=None, verbose=False)
contours_df.loc[:, "group"] = [
    j.split('_')[-1] for j in contours_df.loc[:, "group"]]

# get annotation docs
annprops = {
    'F': (ymax - ymin) / tissue.shape[0],
    'X_OFFSET': xmin,
    'Y_OFFSET': ymin,
    'opacity': 0,
    'lineWidth': 3.0,
}
annotation_docs = get_annotation_documents_from_contours(
    contours_df.copy(), docnamePrefix='spixel', annprops=annprops,
    annots_per_doc=1000, separate_docs_by_group=True,
    verbose=False, monitorPrefix="spixels : annotation docs")
for doc in annotation_docs:
    _ = gc.post(
            "/annotation?itemId=" + slide_id, json=doc)

# %% ==========================================================================

# Visualize contiguous superpixels by cellularity

# get cellularity cluster membership mask
cellularity_mask = np.zeros(spixel_mask.shape)
for spval, sp in fdata.iterrows():
    cellularity_mask[spixel_mask == spval] = sp['cluster']

# Define GTCodes dataframe
GTCodes_df = DataFrame(columns=['group', 'GT_code', 'color'])
for spval, cp in cluster_props.items():
    spstr = 'cellularity-%d' % (cp['cellularity'])
    GTCodes_df.loc[spstr, 'group'] = spstr
    GTCodes_df.loc[spstr, 'GT_code'] = spval
    GTCodes_df.loc[spstr, 'color'] = cp['color']

# get contours df
contours_df = get_contours_from_mask(
    MASK=cellularity_mask, GTCodes_df=GTCodes_df,
    get_roi_contour=False, MIN_SIZE=0, MAX_SIZE=None, verbose=False)

# get annotation docs
annprops = {
    'F': (ymax - ymin) / tissue.shape[0],
    'X_OFFSET': xmin,
    'Y_OFFSET': ymin,
    'opacity': opacity,
    'lineWidth': lineWidth,
}
annotation_docs = get_annotation_documents_from_contours(
    contours_df.copy(), docnamePrefix='spixel', annprops=annprops,
    annots_per_doc=1000, separate_docs_by_group=True,
    verbose=False, monitorPrefix="spixels : annotation docs")
for doc in annotation_docs:
    _ = gc.post(
            "/annotation?itemId=" + slide_id, json=doc)
