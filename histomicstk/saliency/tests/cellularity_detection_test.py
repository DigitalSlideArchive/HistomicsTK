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
#SAMPLE_SLIDE_ID = "5d586d76bd4404c6b1f286ae"

#gc = girder_client.GirderClient(apiUrl=APIURL)
# gc.authenticate(interactive=True)
gc.authenticate(apiKey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')


# %%===========================================================================

# import matplotlib.pylab as plt
# from matplotlib.colors import ListedColormap

from imageio import imread
from pandas import DataFrame, concat
from skimage.color import rgb2gray
from skimage.segmentation import slic
from skimage.transform import resize
from sklearn.mixture import GaussianMixture
from skimage.measure import regionprops
from matplotlib import cm

from histomicstk.utils.general_utils import Base_HTK_Class
from histomicstk.preprocessing.color_conversion import lab_mean_std
from histomicstk.preprocessing.color_normalization import reinhard
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
    """Detect cellular regions in a slides by classifying superpixels.

    This uses Simple Linear Iterative Clustering (SLIC) to get superpixels at
    a low slide magnification to detect cellular regions. The first step of
    this pipeline detects tissue regions (i.e. individual tissue pieces)
    using the get_tissue_mask method of the histomicstk.saliency module. Then,
    each tissue piece is processed separately for accuracy and disk space
    efficiency. It is important to keep in mind that this does NOT rely on a
    tile iterator, but loads the entire tissue region (but NOT the whole slide)
    in memory and passes it on to skimage.segmentation.slic method.

    Once superpixels are segmented, the image is deconvolved and features are
    extracted from the hematoxylin channel. Features include intensity and
    possibly also texture features. Then, a mixed component Gaussian mixture
    model is fit to the features, and median intensity is used to rank
    superpixel clusters by 'cellularity' (since we are working with the
    hematoxylin channel).

    Additional functionality includes contour extraction to get the final
    segmentation boundaries of cellular regions and to visualize them in DSA
    using one's preferred colormap.

    """

    def __init__(self, gc, slide_id, **kwargs):
        """Init Cellularity_Detector_Superpixels object.

        Arguments:
        -----------
        gc : object
            girder client object
        slide_id : str
            girder ID of slide
        verbose : int
            0 - Do not print to screen
            1 - Print only key messages
            2 - Print everything to screen
        monitorPrefix : str
            text to prepend to printed statements
        MAG : float
            magnification at which to detect cellularity
        spixel_size_baseMag : int
            approximate superpixel size at base (scan) magnification
        compactness : float
            compactness parameter for the SLIC method. Higher values result
            in more regular superpixels while smaller values are more likely
            to respect tissue boundaries.
        use_grayscale : bool
            use SLIC

        """
        default_attr = {
            'verbose': 1,
            'monitorPrefix': "",
            'MAG': 3.0,
            'cnorm_params': dict(),
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
            'get_tissue_mask_kwargs': {
                'deconvolve_first': False, 'n_thresholding_steps': 1,
                'sigma': 1.5, 'min_size': 500,
            },
        }
        super(Cellularity_Detector_Superpixels, self).__init__(
            default_attr=default_attr)

        # set attribs
        self.gc = gc
        self.slide_id = slide_id

    # %% =====================================================================

    def set_slide_info(self):
        """Set self.slide_info dict."""
        # This is a presistent dict to store information about slide
        self.slide_info = self.gc.get('item/%s/tiles' % self.slide_id)

        # get tissue mask
        thumbnail_rgb = get_slide_thumbnail(self.gc, self.slide_id)
        self.slide_info['labeled'], _ = get_tissue_mask(
                thumbnail_rgb, self.get_tissue_mask_kwargs)

        # Find size relative to WSI
        self.slide_info['F_tissue'] = self.slide_info[
            'sizeX'] / self.slide_info['labeled'].shape[1]

    # %% =====================================================================

    def set_color_normalization_values(
            self, mu=None, sigma=None, ref_image_path=None, what='main'):
        """read target image and  fetch normalization values"""

        assert (
            all([j is not None for j in (mu, sigma)])
            or ref_image_path is not None), \
            "You must provide mu & sigma values or ref. image to get them."
        assert what in ('thumbnail', 'main')

        if ref_image_path is not None:
            ref_im = np.array(imread(self.ref_image_path, pilmode='RGB'))
            mu, sigma = lab_mean_std(ref_im)

        self.cnorm_params[what] = {'mu': mu, 'sigma': sigma}

    # %% =====================================================================



    # %% =====================================================================

    def get_tissue_rgb_from_small_mask(self, tissue_mask):
        """Load tissue RGB from server for single tissue piece."""
        # find coordinates at scan magnification
        tloc = np.argwhere(tissue_mask)
        ymin, xmin = [
            int(j) for j in np.min(tloc, axis=0) * self.slide_info['F_tissue']]
        ymax, xmax = [
            int(j) for j in np.max(tloc, axis=0) * self.slide_info['F_tissue']]
        # load RGB for this tissue piece at saliency magnification
        getStr = "/item/%s/tiles/region?left=%d&right=%d&top=%d&bottom=%d" % (
            self.slide_id, xmin, xmax, ymin, ymax
            ) + "&magnification=%d" % self.MAG
        resp = self.gc.get(getStr, jsonResp=False)
        tissue = get_image_from_htk_response(resp)
        if 'thumbnail' in self.cnorm_params.keys():
            tissue = np.uint8(reinhard(
                im_src=tissue,
                target_mu=self.cnorm_params['thumbnail']['mu'],
                target_sigma=self.cnorm_params['thumbnail']['sigma']))
        tissue_dict = {
            'rgb': tissue,
            'ymin': ymin, 'xmin': xmin,
            'ymax': ymax, 'xmax': xmax,
        }
        return tissue_dict

    # %% =====================================================================

    def get_superpixel_mask(self, tissue_dict, tissue_mask):
        """Use Simple Linear Iterative Clustering (SLIC) to get superpixels."""
        if self.use_grayscale:
            tissue = rgb2gray(tissue_dict['rgb'])
        else:
            tissue = tissue_dict['rgb']

        spixel_size = self.spixel_size_baseMag * (
            self.MAG / self.slide_info['magnification'])
        n_spixels = int(tissue.shape[0] * tissue.shape[1] / spixel_size)

        # get superpixl mask
        spixel_mask = slic(
            tissue, n_segments=n_spixels, compactness=self.compactness)

        # restrict to tissue mask
        tissue_mask = 0 + tissue_mask
        tissue_mask = tissue_mask[
            int(tissue_dict['ymin'] / self.slide_info['F_tissue']):
            int(tissue_dict['ymax'] / self.slide_info['F_tissue']),
            int(tissue_dict['xmin'] / self.slide_info['F_tissue']):
            int(tissue_dict['xmax'] / self.slide_info['F_tissue'])]
        tmask = resize(
            tissue_mask, output_shape=spixel_mask.shape,
            order=0, preserve_range=True)
        spixel_mask[tmask == 0] = 0

        return spixel_mask

    # %% =====================================================================



# %%===========================================================================
# %%===========================================================================
# %%===========================================================================

# %%

#cds = Cellularity_Detector_Superpixels(gc, SAMPLE_SLIDE_ID)

# %%

a

# %%===========================================================================
# %%===========================================================================
# %%===========================================================================


# %%===========================================================================


#
#unique_tvals = list(set(np.unique(labeled)) - {0, })
#
#tval = unique_tvals[1]
##for tval in unique_tvals:
#
## %%===========================================================================
#
#tissue_mask = labeled == tval


# %%===========================================================================

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
