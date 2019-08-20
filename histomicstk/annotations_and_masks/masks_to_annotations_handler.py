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

# %% =====================================================================


class Conditional_Print(object):

    def __init__(self, verbose=True):
        self.verbose = verbose

    def _print(self, text):
        if self.verbose:
            print(text)

# %% =====================================================================


def get_contours_from_bin_mask(bin_mask):
    """Given a binary mask, get opencv contours.

    Parameters
    -----------
    bin_mask : nd array
        ground truth mask (m,n) - int32 with [0, 1] values.

    Returns
    --------
    dict
        a dictionary with the following keys:
            contour group - the actual contour x,y coordinates.
            heirarchy - contour hierarchy. This contains information about
                how contours relate to each other, in the form:
                [Next, Previous, First_Child, Parent,
                index_relative_to_contour_group]
                The last column is added for convenience and is not part of the
                original opencv output.
            outer_contours - index of contours that do not have a parent, and
                are therefore the outermost most contours. These may have
                children (holes), however.
        See docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
        for more information.

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


def _add_contour_to_df(
        contours_df, mask_shape, conts, cidx, nest_info,
        pad_margin=0, MIN_SIZE=30, MAX_SIZE=None, monitorPrefix=""):
    """Add single contour to dataframe (Internal)."""

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
    xmin, ymin = np.min(cont_outer, axis=0)
    nest_width, nest_height = np.max(
            cont_outer, 0) - np.min(cont_outer, 0)
    ymax = ymin + nest_height
    xmax = xmin + nest_width

    # ignore nests that are too small
    assert ((nest_height > MIN_SIZE) and (nest_width > MIN_SIZE)), \
        "%s: TOO SMALL (%d x %d pixels) -- IGNORED" % (
        monitorPrefix, nest_height, nest_width)

    # ignore extremely large nests -- THESE MAY CAUSE SEGMENTATION FAULTS
    if MAX_SIZE is not None:
        assert ((nest_height < MAX_SIZE) and (nest_width < MAX_SIZE)), \
            "%s: EXTREMELY LARGE NEST (%d x %d pixels) -- IGNORED" % (
                monitorPrefix, nest_height, nest_width)

    # assign bounding box location
    ridx = contours_df.shape[0]
    contours_df.loc[ridx, "group"] = nest_info['group']
    contours_df.loc[ridx, "color"] = nest_info['color']
    contours_df.loc[ridx, "ymin"] = ymin - pad_margin
    contours_df.loc[ridx, "ymax"] = ymax - pad_margin
    contours_df.loc[ridx, "xmin"] = xmin - pad_margin
    contours_df.loc[ridx, "xmax"] = xmax - pad_margin

    # add other properties -- maybe useful later
    contours_df.loc[ridx, "has_holes"] = has_holes
    contours_df.loc[ridx, "touches_edge-top"] = 0 + (
            ymin - pad_margin - 2 < 0)
    contours_df.loc[ridx, "touches_edge-left"] = 0 + (
            xmin - pad_margin - 2 < 0)
    contours_df.loc[ridx, "touches_edge-bottom"] = 0 + (
            ymax + pad_margin + 2 > mask_shape[0])
    contours_df.loc[ridx, "touches_edge-right"] = 0 + (
            xmax + pad_margin + 2 > mask_shape[1])

    # get x and y coordinates in HTK friendly format (string)
    cont_outer = conts['contour_group'][outer_cidx][:, 0, :].copy()
    contours_df.loc[ridx, "coords_x"] = \
        ",".join([str(j - pad_margin) for j in list(cont_outer[:, 0])])
    contours_df.loc[ridx, "coords_y"] = \
        ",".join([str(j - pad_margin) for j in list(cont_outer[:, 1])])

    return contours_df

# %% =====================================================================


def get_contours_from_mask(
        MASK, GTCodes_df, groups_to_get=None, MIN_SIZE=30, MAX_SIZE=None,
        verbose=False, monitorPrefix=""):
    """Parse ground truth mask and gets countours for annotations.

    Parameters
    -----------
    MASK : nd array
        ground truth mask (m,n) where pixel values encode group membership.
    GTCodes_df : pandas Dataframe
        the ground truth codes and information dataframe.
        This is a dataframe that is indexed by the annotation group name and
        has the following columns.

        group: str
            group name of annotation, eg. mostly_tumor.
        GT_code: int
            desired ground truth code (in the mask). Pixels of this value
            belong to corresponding group (class).
        color: str
            rgb format. eg. rgb(255,0,0).
    groups_to_get : None
        if None (default) then all groups (ground truth labels) will be
        extracted. Otherwise pass a list fo strings like ['mostly_tumor',].
    MIN_SIZE : int
        minimum bounding box size of contour
    MAX_SIZE : None
        if not None, int. Maximum bounding box size of contour. Sometimes
        very large contours cause segmentation faults that originate from
        opencv and are not caught by python, causing the python process
        to unexpectedly hault. If you would like to set a maximum size to
        defend against this, a suggested maximum would be 15000.
    verbose : bool
        Print progress to screen?
    monitorPrefix : str
        text to prepend to printed statements

    Returns
    --------
    pandas DataFrame
        contours extracted from input mask. The following columns are output.

        group : str
            annotation group (ground truth label).
        color : str
            annotation color if it were to be posted to DSA.
        is_roi : bool
            whether this annotation is a region of interest boundary
        ymin : int
            minimun y coordinate
        ymax : int
            maximum y coordinate
        xmin : int
            minimum x coordinate
        xmax : int
            maximum x coordinate
        has_holes : bool
            whether this contour has holes
        touches_edge-top : bool
            whether this contour touches top mask edge
        touches_edge-bottom : bool
            whether this contour touches bottom mask edge
        touches_edge-left : bool
            whether this contour touches left mask edge
        touches_edge-right : bool
            whether this contour touches right mask edge
        coords_x : str
            vertix x coordinates comma-separated values
        coords_y
            vertix y coordinated comma-separated values

    """
    cpr = Conditional_Print(verbose=verbose)
    _print = cpr._print

    # pad with zeros to be able to detect edge tumor nests later
    pad_margin = 50
    MASK = np.pad(MASK, pad_margin, 'constant')

    # Go through unique groups one by one -- each group (i.e. GTCode)
    # is extracted separately by binarizing the multi-class mask
    if groups_to_get is None:
        groups_to_get = list(GTCodes_df.index)
    contours_df = DataFrame()

    for nestgroup in groups_to_get:

        bin_mask = 0 + (MASK == GTCodes_df.loc[nestgroup, 'GT_code'])

        if bin_mask.sum() < MIN_SIZE * MIN_SIZE:
            _print("%s: %s: NO NESTS!!" % (monitorPrefix, nestgroup))
            continue

        _print("%s: %s: getting contours" % (monitorPrefix, nestgroup))
        conts = get_contours_from_bin_mask(bin_mask=bin_mask)
        n_tumor_nests = conts['outer_contours'].shape[0]

        # add nest contours
        _print("%s: %s: adding contours" % (monitorPrefix, nestgroup))
        for cidx in range(n_tumor_nests):
            try:
                nestcountStr = "%s: nest %s of %s" % (
                    monitorPrefix, cidx, n_tumor_nests)
                if cidx % 25 == 100:
                    _print(nestcountStr)

                contours_df = _add_contour_to_df(
                    contours_df, mask_shape=bin_mask.shape,
                    conts=conts, cidx=cidx,
                    nest_info=dict(GTCodes_df.loc[nestgroup, :]),
                    pad_margin=pad_margin, MIN_SIZE=MIN_SIZE,
                    MAX_SIZE=MAX_SIZE, monitorPrefix=nestcountStr)

            except AssertionError as e:
                _print(e)
                continue

    return contours_df

# %% =====================================================================


def get_single_annotation_document_from_contours(
        contours_df_slice, docname='default',
        F=1.0, X_OFFSET=0, Y_OFFSET=0, opacity=0.3,
        lineWidth=4.0, verbose=True, monitorPrefix=""):
    """Given dataframe of contours, get annotation document.

    This uses the large_image annotation schema to create an annotation
    document that maybe posted to DSA for viewing using something like:
    resp = gc.post("/annotation?itemId=" + slide_id, json=annotation_doc)
    The annotation schema can be found at:
    github.com/girder/large_image/blob/master/docs/annotations.md .

    Parameters
    -----------
    contours_df_slice : pandas DataFrame
        The following columns are of relevance and must be contained.

        group : str
            annotation group (ground truth label).
        color : str
            annotation color if it were to be posted to DSA.
        coords_x : str
            vertix x coordinates comma-separated values
        coords_y
            vertix y coordinated comma-separated values

    Returns
    --------
    dict
        DSA-style annotation document.

    """
    cpr = Conditional_Print(verbose=verbose)
    _print = cpr._print

    def _get_fillColor(lineColor):
        fillColor = lineColor.replace("rgb", "rgba")
        return fillColor[:fillColor.rfind(")")] + ",%.1f)" % opacity

    # Init annotation document in DSA style
    annotation_doc = {'name': docname, 'description': '', 'elements': []}

    # go through nests
    nno = 0
    nnests = contours_df_slice.shape[0]
    for _, nest in contours_df_slice.iterrows():

        nno += 1
        nestStr = "%s: nest %d of %s" % (monitorPrefix, nno, nnests)
        _print(nestStr)

        # Parse coordinates
        try:
            x_coords = F * np.int32(
                [int(j) for j in nest['coords_x'].split(',')]) + X_OFFSET
            y_coords = F * np.int32(
                [int(j) for j in nest['coords_y'].split(',')]) + Y_OFFSET
            zeros = np.zeros(x_coords.shape, dtype=np.int32)
            coords = np.concatenate(
                (x_coords[:, None], y_coords[:, None], zeros[:, None]),
                axis=1)
            coords = coords.tolist()
            coords.append(coords[0])
        except Exception as e:
            _print("%s: ERROR (below) - moving on!!!" % nestStr)
            _print(e)
            continue

        # assign to annotation style. See:
        # github.com/girder/large_image/blob/master/docs/annotations.md
        annotation_style = {
            "group": nest['group'],
            "type": "polyline",
            "lineColor": nest['color'],
            "lineWidth": lineWidth,
            "closed": True,
            "points": coords,
            "label": {'value': nest['group']},
        }
        if opacity > 0:
            annotation_style["fillColor"] = _get_fillColor(nest['color'])

        # append to document
        annotation_doc['elements'].append(annotation_style)

    return annotation_doc

# %% =====================================================================


def get_annotation_documents_from_contours(
        contours_df, ANNOTS_PER_DOC=200, docnamePrefix='default',
        annprops=None, verbose=True, monitorPrefix=""):
    """Given dataframe of contours, get list of annotation documents.

    Parameters
    -----------
    contours_df : pandas DataFrame

    Returns
    --------
    list of dicts
        DSA-style annotation document.

    """
    if annprops is None:
        annprops = {
            'F': 1.0,
            'X_OFFSET': 0,
            'Y_OFFSET': 0,
            'opacity': 0.3,
            'lineWidth': 4.0,
        }

    # Go through documents -- add every N annotations to a separate document
    if contours_df.shape[0] > ANNOTS_PER_DOC:
        docbounds = list(range(0, contours_df.shape[0], ANNOTS_PER_DOC))
        docbounds[-1] = contours_df.shape[0]
    else:
        docbounds = [0, contours_df.shape[0]]

    annotation_docs = []
    for docidx in range(len(docbounds)-1):
        docStr = "%s: doc %d of %d" % (
                monitorPrefix, docidx+1, len(docbounds)-1)
        start = docbounds[docidx]
        end = docbounds[docidx+1]
        contours_df_slice = contours_df.iloc[start:end, :]
        annotation_doc = get_single_annotation_document_from_contours(
            contours_df_slice, docname="%s-%d" % (docnamePrefix, docidx),
            verbose=verbose, monitorPrefix=docStr, **annprops)
        if len(annotation_doc['elements']) > 0:
            annotation_docs.append(annotation_doc)

    return annotation_docs

# %% =====================================================================


# get specified (or any) contours from mask
groups_to_get = [
    'mostly_tumor', 'mostly_stroma', 'mostly_lymphocytic_infiltrate']
contours_df = get_contours_from_mask(
    MASK=MASK, GTCodes_df=GTCodes_df, groups_to_get=groups_to_get,
    MIN_SIZE=30, MAX_SIZE=None,
    verbose=True, monitorPrefix=MASKNAME[:12])

# get list of annotation documents
annotation_docs = get_annotation_documents_from_contours(
    contours_df, ANNOTS_PER_DOC=10, docnamePrefix='test',
    annprops=None, verbose=True, monitorPrefix="")
