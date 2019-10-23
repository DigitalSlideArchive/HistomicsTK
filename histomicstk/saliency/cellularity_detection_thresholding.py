#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 02:37:52 2019

@author: mtageld
"""
import numpy as np
from imageio import imread
from pandas import DataFrame, concat, read_csv
#from skimage.color import rgb2gray
#from skimage.segmentation import slic
from skimage.transform import resize
#from sklearn.mixture import GaussianMixture
#from skimage.measure import regionprops
#from matplotlib import cm
from PIL import Image

from histomicstk.utils.general_utils import Base_HTK_Class
from histomicstk.preprocessing.color_conversion import lab_mean_std
from histomicstk.preprocessing.color_conversion import rgb_to_hsi
from histomicstk.preprocessing.color_conversion import rgb_to_lab
from histomicstk.preprocessing.color_normalization import (
    reinhard, deconvolution_based_normalization)
from histomicstk.preprocessing.color_deconvolution import (
    rgb_separate_stains_macenko_pca, _reorder_stains)
from histomicstk.saliency.tissue_detection import (
    get_slide_thumbnail, get_tissue_mask,
    get_tissue_boundary_annotation_documents,
    threshold_multichannel, _get_largest_regions)
from histomicstk.features.compute_intensity_features import (
    compute_intensity_features)
#from histomicstk.annotations_and_masks.masks_to_annotations_handler import (
#    get_contours_from_mask, get_annotation_documents_from_contours)
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    get_image_from_htk_response)
#from histomicstk.features.compute_intensity_features import (
#    compute_intensity_features)
#from histomicstk.features.compute_haralick_features import (
#    compute_haralick_features)


Image.MAX_IMAGE_PIXELS = None

# %%===========================================================================
# =============================================================================


class CDT_single_tissue_piece(object):
    """Detect various regions in a single tissue piece (internal)."""

    def __init__(self, cdt, tissue_mask, monitorPrefix=""):
        """Detect whitespace, saliency, etc in one tissue piece (Internal).

        Arguments
        ----------
        cdt : object
            Cellularity_detector_thresholding instance
        tissue_mask : np array
            (m x n) mask of the tissue piece at cdt.MAG magnification
        monitorPrefix : str
            Text to prepend to printed statements

        """
        self.cdt = cdt
        self.tissue_mask = 0 + tissue_mask
        self.monitorPrefix = monitorPrefix

    # =========================================================================

    def run(self):
        """Get cellularity and optionally visualize on DSA."""
        self.restrict_mask_to_single_tissue_piece()
        self.cdt._print2("%s: set_tissue_rgb()" % self.monitorPrefix)
        self.set_tissue_rgb()
        self.cdt._print2("%s: initialize_labeled_mask()" % self.monitorPrefix)
        self.initialize_labeled_mask()
        self.cdt._print2(
            "%s: assign_components_by_thresholding()" % self.monitorPrefix)
        self.assign_components_by_thresholding()
        self.cdt._print2(
            "%s: color_normalize_unspecified_components()"
            % self.monitorPrefix)
        self.color_normalize_unspecified_components()

    # =========================================================================

    def restrict_mask_to_single_tissue_piece(self):
        """Only keep relevant part of slide mask."""
        # find coordinates at scan magnification
        tloc = np.argwhere(self.tissue_mask)
        F = self.cdt.slide_info['F_tissue']
        self.ymin, self.xmin = [int(j) for j in np.min(tloc, axis=0) * F]
        self.ymax, self.xmax = [int(j) for j in np.max(tloc, axis=0) * F]
        self.tissue_mask = self.tissue_mask[
            int(self.ymin / F): int(self.ymax / F),
            int(self.xmin / F): int(self.xmax / F)]

    # =========================================================================

    def set_tissue_rgb(self):
        """Load RGB from server for single tissue piece."""
        # load RGB for this tissue piece at saliency magnification
        getStr = "/item/%s/tiles/region?left=%d&right=%d&top=%d&bottom=%d" % (
            self.cdt.slide_id, self.xmin, self.xmax, self.ymin, self.ymax
            ) + "&magnification=%d" % self.cdt.MAG
        resp = self.cdt.gc.get(getStr, jsonResp=False)
        self.tissue_rgb = get_image_from_htk_response(resp)

    # =========================================================================

    def initialize_labeled_mask(self):
        """"Placeholder."""
        # resize tissue mask to target mag
        self.labeled = resize(
            self.tissue_mask, output_shape=self.tissue_rgb.shape[:2],
            order=0, preserve_range=True)
        self.labeled[self.labeled > 0] = self.cdt.GTcodes.loc[
            'not_specified', 'GT_code']

    # =========================================================================

    def assign_components_by_thresholding(self):
        """Placeholder."""
        # get HSI and LAB images
        self.cdt._print2(
            "%s: -- get HSI and LAB images ..." % self.monitorPrefix)
        tissue_hsi = rgb_to_hsi(self.tissue_rgb)
        tissue_lab = rgb_to_lab(self.tissue_rgb)

        # extract components using HSI/LAB thresholds

        hsi_components = self.cdt.hsi_thresholds.keys()
        lab_components = self.cdt.lab_thresholds.keys()

        for component in self.cdt.ordered_components:

            self.cdt._print2("%s: -- thresholding %s ..." % (
                self.monitorPrefix, component))

            if component in hsi_components:
                lab, _ = threshold_multichannel(
                    tissue_hsi,
                    channels=['hue', 'saturation', 'intensity'],
                    thresholds=self.cdt.hsi_thresholds[component],
                    just_threshold=False,
                    get_tissue_mask_kwargs=self.cdt.get_tissue_mask_kwargs2)
            elif component in lab_components:
                lab, _ = threshold_multichannel(
                    tissue_lab,
                    channels=['l', 'a', 'b'],
                    thresholds=self.cdt.lab_thresholds[component],
                    just_threshold=True,
                    get_tissue_mask_kwargs=self.cdt.get_tissue_mask_kwargs2)
            else:
                raise ValueError("Unknown component name.")

            lab[self.labeled == 0] = 0  # restrict to tissue mask
            self.labeled[lab > 0] = self.cdt.GTcodes.loc[component, 'GT_code']

        # This deals with holes in tissue
        self.labeled[self.labeled == 0] = self.cdt.GTcodes.loc[
            'outside_tissue', 'GT_code']

    # =========================================================================

    def color_normalize_unspecified_components(self):
        """"Placeholder."""
        if self.cdt.color_normalization_method == 'reinhard':
            self.tissue_rgb = reinhard(
                self.tissue_rgb,
                target_mu=self.cdt.target_stats_reinhard['mu'],
                target_sigma=self.cdt.target_stats_reinhard['sigma'],
                mask_out=self.labeled != GTcodes
                    .loc["not_specified", "GT_code"])

        elif self.cdt.color_normalization_method == 'macenko_pca':
            self.tissue_rgb = deconvolution_based_normalization(
                self.tissue_rgb, W_target=self.cdt.target_W_macenko,
                mask_out=self.labeled != GTcodes
                    .loc["not_specified", "GT_code"],
                stain_unmixing_routine_params={
                    'stains': ['hematoxylin', 'eosin'],
                    'stain_unmixing_method': 'macenko_pca',
                    }, )
        else:
            pass

    # =========================================================================


# %%===========================================================================
# =============================================================================


class Cellularity_detector_thresholding(Base_HTK_Class):
    """Placeholder."""

    def __init__(self, gc, slide_id, GTcodes, **kwargs):
        """Placeholder."""
        default_attr = {

            # The following are already assigned defaults by Base_HTK_Class
            # 'verbose': 1,
            # 'monitorPrefix': "",
            # 'logging_savepath': None,
            # 'suppress_warnings': False,

            'MAG': 3.0,

            # Must be in ['reinhard', 'macenko_pca', 'none']
            'color_normalization_method': 'macenko_pca',

            # TCGA-A2-A3XS-DX1_xmin21421_ymin37486_.png, Amgad et al, 2019)
            # is used as the target image for reinhard & macenko normalization

            # for macenco (obtained using rgb_separate_stains_macenko_pca()
            # and using reordered such that columns are the order:
            # Hamtoxylin, Eosin, Null
            'target_W_macenko': np.array([
                [0.5807549,  0.08314027,  0.08213795],
                [0.71681094,  0.90081588,  0.41999816],
                [0.38588316,  0.42616716, -0.90380025]
            ]),

            # TCGA-A2-A3XS-DX1_xmin21421_ymin37486_.png, Amgad et al, 2019)
            # Reinhard color norm. standard
            'target_stats_reinhard': {
                'mu': np.array([8.74108109, -0.12440419,  0.0444982]),
                'sigma': np.array([0.6135447, 0.10989545, 0.0286032]),
            },

            # kwargs for getting masks for all tissue pieces (thumbnail)
            'get_tissue_mask_kwargs': {
                'deconvolve_first': True, 'n_thresholding_steps': 1,
                'sigma': 1.5, 'min_size': 500,
            },

            # components to extract by HSI thresholding
            'keep_components': ['blue_sharpie', 'blood', 'whitespace', ],

            # kwargs for getting components masks
            'get_tissue_mask_kwargs2': {
                'deconvolve_first': False, 'n_thresholding_steps': 1,
                'sigma': 5.0, 'min_size': 50,
            },

            # min/max thresholds for HSI and LAB
            'hsi_thresholds': {
                'whitespace': {
                    'hue': {'min': 0, 'max': 1.0},
                    'saturation': {'min': 0, 'max': 0.2},
                    'intensity': {'min': 220, 'max': 255},
                },
            },
            'lab_thresholds': {
                'blue_sharpie': {
                    'l': {'min': -1000, 'max': 1000},
                    'a': {'min': -1000, 'max': 1000},
                    'b': {'min': -1000, 'max': -0.02},
                },
                'blood': {
                    'l': {'min': -1000, 'max': 1000},
                    'a': {'min': 0.02, 'max': 1000},
                    'b': {'min': -1000, 'max': 1000},
                },
            },
        }
        default_attr.update(kwargs)
        super(Cellularity_detector_thresholding, self).__init__(
            default_attr=default_attr)

        self.color_normalization_method = \
            self.color_normalization_method.lower()
        assert self.color_normalization_method in [
            'reinhard', 'macenko_pca', 'none']

        # set attribs
        self.gc = gc
        self.slide_id = slide_id
        self.GTcodes = GTcodes

        self.fix_GTcodes()

    # %% ======================================================================

    def fix_GTcodes(self):
        """Placeholder."""
        self.GTcodes.sort_values('overlay_order', axis=0, inplace=True)
        self.GTcodes.index = self.GTcodes.loc[:, "group"]
        self.ordered_components = list(self.GTcodes.loc[:, "group"])

        # only keep relevant components
        for c in self.ordered_components.copy():
            if c not in self.keep_components:
                self.ordered_components.remove(c)

    # %% ======================================================================

    def run(self):
        """Placeholder."""
        # get mask, each unique value is a single tissue piece
        self._print1(
            "%s: set_slide_info_and_get_tissue_mask()" % self.monitorPrefix)
        labeled = self.set_slide_info_and_get_tissue_mask()

        # Go through tissue pieces and do run sequence
        unique_tvals = list(set(np.unique(labeled)) - {0, })
        tissue_pieces = [None for _ in range(len(unique_tvals))]
        for idx, tval in enumerate(unique_tvals):
            monitorPrefix = "%s: Tissue piece %d of %d" % (
                self.monitorPrefix, idx+1, len(unique_tvals))
            self._print1(monitorPrefix)
            tissue_pieces[idx] = CDT_single_tissue_piece(
                self, tissue_mask=labeled == tval, monitorPrefix=monitorPrefix)
            tissue_pieces[idx].run()
            del tissue_pieces[idx].tissue_rgb  # too much space

        return tissue_pieces

    # %% ======================================================================

    def set_color_normalization_target(
            self, ref_image_path, color_normalization_method='macenko_pca'):
        """Set color normalization values to use from target image."""

        # read input image
        ref_im = np.array(imread(ref_image_path, pilmode='RGB'))

        # assign target values

        color_normalization_method = color_normalization_method.lower()

        if color_normalization_method == 'reinhard':
            mu, sigma = lab_mean_std(ref_im)
            self.target_stats_reinhard['mu'] = mu
            self.target_stats_reinhard['sigma'] = sigma

        elif color_normalization_method == 'macenko_pca':
            self.target_W_macenko = _reorder_stains(
                rgb_separate_stains_macenko_pca(ref_im, I_0=None),
                stains=['hematoxylin', 'eosin'])
        else:
            raise ValueError(
                "Unknown color_normalization_method: %s" %
                (color_normalization_method))

        self.color_normalization_method = color_normalization_method

    # %% ======================================================================

    def set_slide_info_and_get_tissue_mask(self):
        """Set self.slide_info dict and self.labeled tissue mask."""
        # This is a presistent dict to store information about slide
        self.slide_info = self.gc.get('item/%s/tiles' % self.slide_id)

        # get tissue mask
        thumbnail_rgb = get_slide_thumbnail(self.gc, self.slide_id)

        # get labeled tissue mask -- each unique value is one tissue piece
        labeled, _ = get_tissue_mask(
            thumbnail_rgb, **self.get_tissue_mask_kwargs)

        if len(np.unique(labeled)) < 2:
            raise ValueError("No tissue detected!")

        # Find size relative to WSI
        self.slide_info['F_tissue'] = self.slide_info[
            'sizeX'] / labeled.shape[1]

        return labeled

# %%===========================================================================

# %%===========================================================================
# =============================================================================

import girder_client

# %%

# Constants & prep work

APIURL = 'http://candygram.neurology.emory.edu:8080/api/v1/'
SAMPLE_SLIDE_ID = "5d817f5abd4404c6b1f744bb"

gc = girder_client.GirderClient(apiUrl=APIURL)
# gc.authenticate(interactive=True)
gc.authenticate(apiKey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')

# read GT codes dataframe
GTcodes = read_csv('./tests/saliency_GTcodes.csv')

# %%

cdt = Cellularity_detector_thresholding(
    gc, slide_id=SAMPLE_SLIDE_ID, GTcodes=GTcodes, verbose=2)

self = cdt
# %%







