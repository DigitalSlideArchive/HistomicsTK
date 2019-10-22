#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 02:37:52 2019

@author: mtageld
"""
import numpy as np
from imageio import imread
#from pandas import DataFrame, concat
#from skimage.color import rgb2gray
#from skimage.segmentation import slic
#from skimage.transform import resize
#from sklearn.mixture import GaussianMixture
#from skimage.measure import regionprops
#from matplotlib import cm
from PIL import Image

from histomicstk.utils.general_utils import Base_HTK_Class
from histomicstk.preprocessing.color_conversion import lab_mean_std
#from histomicstk.preprocessing.color_normalization import reinhard
from histomicstk.saliency.tissue_detection import (
    get_slide_thumbnail, get_tissue_mask,
    get_tissue_boundary_annotation_documents)
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
        # get tissue rgb and mask
        self.restrict_mask_to_single_tissue_piece()
        self.cdt._print2("%s: set_tissue_rgb()" % self.monitorPrefix)
        self.set_tissue_rgb()

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

        # TODO -- color normalization proper
        # color normalization if desired
        if 'main' in self.cdt.cnorm_params.keys():
            raise Exception("NOT IMPLEMENTED YET!")

    # =========================================================================


# %%===========================================================================
# =============================================================================


class Cellularity_detector_thresholding(Base_HTK_Class):
    """Placeholder."""

    def __init__(self, gc, slide_id, **kwargs):
        """Placeholder."""
        default_attr = {

            # The following are already assigned defaults by Base_HTK_Class
            # 'verbose': 1,
            # 'monitorPrefix': "",
            # 'logging_savepath': None,
            # 'suppress_warnings': False,

            'cnorm_params': dict(),
            'MAG': 3.0,
            'get_tissue_mask_kwargs': {
                'deconvolve_first': True, 'n_thresholding_steps': 1,
                'sigma': 1.5, 'min_size': 500,
            },
        }
        default_attr.update(kwargs)
        super(Cellularity_detector_thresholding, self).__init__(
            default_attr=default_attr)

        # set attribs
        self.gc = gc
        self.slide_id = slide_id

    # %% ======================================================================

    def run(self):
        """Placeholder."""
        if (len(self.cnorm_params) == 0) and (not self.suppress_warnings):
            input("""
                %s: WARNING!! Consider running set_color_normalization_values()
                first, using what='thumbnail' and/or what='main' before running
                this method. Continue anyway?""" % self.monitorPrefix)

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

    def set_color_normalization_values(
            self, mu=None, sigma=None, ref_image_path=None, what='main'):
        """Set color normalization values for thumbnail or main image."""

        # TODO -- incorporate masking and macenko!!
        raise Exception("[WORK IN PROGRESS!!!!!!]")

        assert (
            all([j is not None for j in (mu, sigma)])
            or ref_image_path is not None), \
            "You must provide mu & sigma values or ref. image to get them."
        assert what in ('thumbnail', 'main')

        if ref_image_path is not None:
            ref_im = np.array(imread(ref_image_path, pilmode='RGB'))
            mu, sigma = lab_mean_std(ref_im)

        self.cnorm_params[what] = {'mu': mu, 'sigma': sigma}

    # %% ======================================================================

    def set_slide_info_and_get_tissue_mask(self):
        """Set self.slide_info dict and self.labeled tissue mask."""
        # This is a presistent dict to store information about slide
        self.slide_info = self.gc.get('item/%s/tiles' % self.slide_id)

        # get tissue mask
        thumbnail_rgb = get_slide_thumbnail(self.gc, self.slide_id)

        # color normalization if desired
        if 'thumbnail' in self.cnorm_params.keys():
            # TODO -- reinhard normalization of thumbnail!!
            pass

        # get labeled tissue mask -- each unique value is one tissue piece
        labeled, _ = get_tissue_mask(
            thumbnail_rgb, **self.get_tissue_mask_kwargs)

        if len(np.unique(labeled)) < 2:
            raise ValueError("No tissue detected!")

        if self.visualize_tissue_boundary:
            annotation_docs = get_tissue_boundary_annotation_documents(
                self.gc, slide_id=self.slide_id, labeled=labeled)
            for doc in annotation_docs:
                _ = self.gc.post(
                    "/annotation?itemId=" + self.slide_id, json=doc)

        # Find size relative to WSI
        self.slide_info['F_tissue'] = self.slide_info[
            'sizeX'] / labeled.shape[1]

        return labeled

# %%===========================================================================
