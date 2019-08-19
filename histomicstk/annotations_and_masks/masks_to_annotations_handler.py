# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 18:33:48 2019.

@author: tageldim

"""

import os
import numpy as np
from pandas import DataFrame, read_csv
from imageio import imread
import cv2
# from shapely.geometry.polygon import Polygon

# from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
#     # from annotation_and_mask_utils import (
#     get_bboxes_from_slide_annotations, _get_idxs_for_all_rois,
#     get_idxs_for_annots_overlapping_roi_by_bbox, _get_element_mask,
#     _get_and_add_element_to_roi)

# %% =====================================================================

# read GTCodes dataframe
GTCODE_PATH = os.path.join(
    os.getcwd(), '..', '..', 'plugin_tests',
    'test_files', 'sample_GTcodes.csv')
GTCodes_df = read_csv(GTCODE_PATH)
GTCodes_df.index = GTCodes_df.loc[:, 'group']

# read mask
MASKNAME = "TCGA-A2-A0YE-01Z-00-DX1.8A2E3094-5755-42BC-969D-7F0A2ECA0F39" + \
    "_left-59206_top-33505_mag-BASE.png"
MASKPATH = os.path.join(os.getcwd(), '..', '..', '..', 'Masks', MASKNAME)
MASK = imread(MASKPATH)

# get specified (or any) contours from mask
groups_to_get = [
    'mostly_tumor', 'mostly_stroma', 'mostly_lymphocytic_infiltrate']
nest_contours = get_contours_from_mask(
    MASK=MASK, GTCodes_df=GTCodes_df, groups_to_get=groups_to_get,
    MIN_SIZE=30, MAX_SIZE=None,
    verbose=True, monitorPrefix=MASKNAME[:12])

# %% =====================================================================


def get_contours_from_bin_mask(bin_mask):
    """Given a binary mask, get opencv contours.

    Parameters
    -----------
    bin_mask : nd array
        ground truth mask (m,n) - bool

    Returns
    --------
    dict
        contour group, heirarchy, outer_contours

    """
    # Get contours using openCV. See this for modes:
    # https://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    # we use the flag RETR_CCOMP so that we get boundary as well as holes
    # hierearchy output is: [Next, Previous, First_Child, Parent]
    ROI_cvuint8 = cv2.convertScaleAbs(bin_mask)
    cvout = cv2.findContours(
        ROI_cvuint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if len(cvout) < 3:
        contour_group, hierarchy = cvout[0], cvout[1]
    else:
        contour_group, hierarchy = cvout[1], cvout[2]
    hierarchy = hierarchy[0, ...]
    # outermost contours are those that don't have a parent
    # We'll append an index column to the rightmost end
    # to keep track of things better relative to contour_group, now it is:
    # [Next, Previous, First_Child, Parent, index_relative_to_contour_group]
    hierarchy = np.concatenate((hierarchy, np.arange(
            hierarchy.shape[0])[..., None]), axis=1)
    outer_contours = hierarchy[hierarchy[:, 3] == -1, :]
    conts = {
        'contour_group': contour_group,
        'hierarchy': hierarchy,
        'outer_contours': outer_contours,
    }
    return conts

# %% =====================================================================


def _add_nest_contour(
        nest_contours, mask_shape, conts, cidx, classlabel,
        pad_margin=0, MIN_SIZE=30, MAX_SIZE=None, monitorPrefix=""):
    """Add single nest contour to dataframe (Internal)."""

    # get coordinates for this contour. These are in x,y format.
    outer_cidx = conts['outer_contours'][cidx, 4]
    cont_outer = conts['contour_group'][outer_cidx][:, 0, :]
    assert cont_outer.shape[0] > 10, \
        "%s: TOO SIMPLE (%d coordinates) -- IGNORED" % (
        monitorPrefix, cont_outer.shape[0])

    # Get index of first child (hole)
    inner_cidx = conts['outer_contours'][cidx, 2]
    has_holes = 0 + (inner_cidx > -1)

    # get nest location and size
    cmin, rmin = np.min(cont_outer, axis=0)
    nest_width, nest_height = np.max(
            cont_outer, 0) - np.min(cont_outer, 0)
    rmax = rmin + nest_height
    cmax = cmin + nest_width

    # ignore nests that are too small
    assert ((nest_height > MIN_SIZE) and (nest_width > MIN_SIZE)), \
        "%s: TOO SMALL (%d x %d pixels) -- IGNORED" % (
        monitorPrefix, nest_height, nest_width)

    # ignore extremely large nests -- THESE CAUSE SEGMENTATION FAULTS
    if MAX_SIZE is not None:
        assert ((nest_height < MAX_SIZE) and (nest_width < MAX_SIZE)), \
            "%s: EXTREMELY LARGE NEST (%d x %d pixels) -- IGNORED" % (
                monitorPrefix, nest_height, nest_width)

    # assign bounding box location
    ridx = nest_contours.shape[0]
    nest_contours.loc[ridx, "label"] = classlabel
    nest_contours.loc[ridx, "rmin"] = rmin - pad_margin
    nest_contours.loc[ridx, "rmax"] = rmax - pad_margin
    nest_contours.loc[ridx, "cmin"] = cmin - pad_margin
    nest_contours.loc[ridx, "cmax"] = cmax - pad_margin

    # add other properties useful later
    nest_contours.loc[ridx, "has_holes"] = has_holes
    nest_contours.loc[ridx, "touches_edge-top"] = 0 + (
            rmin - pad_margin - 2 < 0)
    nest_contours.loc[ridx, "touches_edge-left"] = 0 + (
            cmin - pad_margin - 2 < 0)
    nest_contours.loc[ridx, "touches_edge-bottom"] = 0 + (
            rmax + pad_margin + 2 > mask_shape[0])
    nest_contours.loc[ridx, "touches_edge-right"] = 0 + (
            cmax + pad_margin + 2 > mask_shape[1])

    # get x and y coordinates in HTK friendly format
    cont_outer = conts['contour_group'][outer_cidx][:, 0, :].copy()
    nest_contours.loc[ridx, "coords_x"] = \
        ",".join([str(j - pad_margin) for j in list(cont_outer[:, 0])])
    nest_contours.loc[ridx, "coords_y"] = \
        ",".join([str(j - pad_margin) for j in list(cont_outer[:, 1])])

    return nest_contours

# %% =====================================================================


def get_contours_from_mask(
        MASK, GTCodes_df, groups_to_get=None, MIN_SIZE=30, MAX_SIZE=None,
        verbose=False, monitorPrefix=""):
    """Parse ground truth mask and gets countours for annotations.

    Parameters
    -----------
    MASK : nd array
        ground truth mask (m,n)

    Returns
    --------
    pandas DataFrame
        contours for annotations

    """
    def _print(text):
        if verbose:
            print(text)

    # pad with zeros to be able to detect edge tumor nests later
    pad_margin = 50
    MASK = np.pad(MASK, pad_margin, 'constant')

    # Go through unique groups one by one -- each group (i.e. GTCode)
    # is extracted separately by binarizing the multi-class mask
    if groups_to_get is None:
        groups_to_get = list(GTCodes_df.index)
    nest_contours = DataFrame()

    for classlabel in groups_to_get:

        bin_mask = 0 + (MASK == GTCodes_df.loc[classlabel, 'GT_code'])

        if bin_mask.sum() < MIN_SIZE * MIN_SIZE:
            _print("%s: %s: NO NESTS!!" % (monitorPrefix, classlabel))
            continue

        _print("%s: %s: getting contours" % (monitorPrefix, classlabel))
        conts = get_contours_from_bin_mask(bin_mask=bin_mask)
        n_tumor_nests = conts['outer_contours'].shape[0]

        # add nest contours
        _print("%s: %s: adding contours" % (monitorPrefix, classlabel))
        for cidx in range(n_tumor_nests):
            try:
                nestcountStr = "%s: nest %s of %s" % (
                    monitorPrefix, cidx, n_tumor_nests)
                if cidx % 25 == 100:
                    _print(nestcountStr)

                nest_contours = _add_nest_contour(
                    nest_contours, mask_shape=bin_mask.shape,
                    conts=conts, cidx=cidx, classlabel=classlabel,
                    pad_margin=pad_margin, MIN_SIZE=MIN_SIZE,
                    MAX_SIZE=MAX_SIZE, monitorPrefix=nestcountStr)

            except AssertionError as e:
                _print(e)
                continue

    return nest_contours

# %% =====================================================================
