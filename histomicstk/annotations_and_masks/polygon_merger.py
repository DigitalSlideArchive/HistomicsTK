# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 16:25:06 2019

@author: tageldim
"""

import numpy as np
from pandas import DataFrame, concat
import cv2
from shapely.geometry.polygon import Polygon
from PIL import Image
from imageio import imread
from masks_to_annotations_handler import get_contours_from_mask

# %% =====================================================================


class Conditional_Print(object):
    """Print to screen if certain conditions are satisfied (Internal)."""

    def __init__(self, verbose=True):
        """Init. This is for PEP compliance."""
        self.verbose = verbose

    def _print(self, text):
        if self.verbose:
            print(text)

# %% =====================================================================


def get_contours_from_all_masks(maskpaths, contkwargs):
    """Get contours_df from all masks.

    This is a wrapper around get_contours_from_mask(), with the added
    functionality of separating out contorus at roi edge from those that
    are not.

    Arguments:
    -----------
    maskpaths : list
        list of strings representing pathos to masks
    contkwargs : dict
        dictionary of kwargs to pass to get_contours_from_mask()

    Returns:
    ---------
    dict
        indexed by maskname, each entry is a contours dataframe

    """

    ordinary_contours = dict()
    edge_contours = dict()

    for midx, maskpath in enumerate(maskpaths):

        # extract contours
        MASK = imread(maskpath)
        contours_df = get_contours_from_mask(
            MASK=MASK, monitorPrefix="mask %d of %d" % (midx, len(maskpaths)),
            **contkwargs)

        # separate edge from non-edge contours
        edgeids = []
        for edge in ['top', 'left', 'bottom', 'right']:
            edgeids.extend(list(contours_df.loc[contours_df.loc[
                :, 'touches_edge-%s' % edge] == 1, :].index))
        edgeids = list(set(edgeids))
        roiname = os.path.split(maskpath)[1]
        edge_contours[roiname] = contours_df.loc[edgeids, :].copy()
        ordinary_contours[roiname] = contours_df.drop(edgeids, axis=0)

    return ordinary_contours, edge_contours

# %% =====================================================================


def _get_mask_offsets_from_masknames(maskpaths):
    """Get dictionary of mask offsets (top and left) (Internal).

    The pattern _left-123_ and _top-123_ is assumed to
    encode the x and y offset of the mask at base magnification.

    Arguments:
    -----------
    maskpaths : list
        names of masks (list of str)

    Returns:
    ----------
    dict
        indexed by maskname, each entry is a dict with keys top and left.

    """
    roi_offsets = dict()
    for maskpath in maskpaths:
        maskname = os.path.split(maskpath)[1]
        roi_offsets[maskname] = {
            'left': int(maskname.split('_left-')[1].split('_')[0]),
            'top': int(maskname.split('_top-')[1].split('_')[0]),
        }
    return roi_offsets

# %% =====================================================================


def get_roi_bboxes(maskpaths, roi_offsets=None):
    """Get dictionary of roi bounding boxes.

    Arguments:
    -----------
    maskpaths : list
        names of masks (list of str)
    roi_offsets : dict (default, None)
        dict indexed by maskname, each entry is a dict with keys
        top and left each is an integer. If None, then the pattern
        _left-123_ and _top-123_ is assumed to encode the x and y
        offset of the mask (i.e. inferred from mask name)

    Returns:
    ----------
    dict
        dict indexed by maskname, each entry is a dict with keys
        top, left, bottom, right, all of which are integers

    """
    if roi_offsets is not None:
        roiinfos = roi_offsets.copy()
    else:
        # get offset for all rois. This result is a dict that is indexed
        # by maskname, each entry is a dict with keys 'top' and 'left'.
        roiinfos = _get_mask_offsets_from_masknames(maskpaths)

    for maskpath in maskpaths:
        # Note: the following method does NOT actually load the mask
        # but just uses pillow to get its metadata. See:
        # https://stackoverflow.com/questions/15800704/ ...
        # ... get-image-size-without-loading-image-into-memory
        mask_obj = Image.open(maskpath, mode='r')
        width, height = mask_obj.size
        maskname = os.path.split(maskpath)[1]
        roiinfos[maskname]['right'] = roiinfos[maskname]['left'] + width
        roiinfos[maskname]['bottom'] = roiinfos[maskname]['top'] + height

    return roiinfos

# %% =====================================================================


def _get_roi_pairs(roinames):
    """Get unique roi pairs (Internal)."""
    ut = np.triu_indices(len(roinames), k=1)
    roi_pairs = []
    for pairidx in range(len(ut[0])):
        roi_pairs.append((ut[0][pairidx], ut[1][pairidx]))
    return roi_pairs

# %% =====================================================================


def _get_shared_roi_edges(roiinfos):
    """Get shared edges between rois in same slide (Internal)."""
    roinames = list(roiinfos.keys())
    edgepairs = [
        ('left', 'right'), ('right', 'left'),
        ('top', 'bottom'), ('bottom', 'top'),
    ]
    roi_pairs = _get_roi_pairs(roinames)

    # init shared edges
    shared_edges = DataFrame(columns=[
            'roi1-name', 'roi1-edge', 'roi2-name', 'roi2-edge'])

    for roi_pair in roi_pairs:
        roi1name = roinames[roi_pair[0]]
        roi2name = roinames[roi_pair[1]]
        idx = shared_edges.shape[0]
        for edgepair in edgepairs:
            # check if they share bounds for one edge
            if np.abs(
                  roiinfos[roi1name][edgepair[0]]
                  - roiinfos[roi2name][edgepair[1]]) < 2:
                # ensure they overlap in location along other axis
                if 'left' in edgepair:
                    start, end = ('top', 'bottom')
                else:
                    start, end = ('left', 'right')
                realStart = np.min(
                    (roiinfos[roi1name][start], roiinfos[roi2name][start]))
                realEnd = np.max(
                    (roiinfos[roi1name][end], roiinfos[roi2name][end]))
                length = realEnd - realStart
                nonoverlap_length = (
                    roiinfos[roi1name][end] - roiinfos[roi1name][start]) + (
                    roiinfos[roi2name][end] - roiinfos[roi2name][start])
                if length < nonoverlap_length:
                    shared_edges.loc[idx, 'roi1-name'] = roi1name
                    shared_edges.loc[idx, 'roi1-edge'] = edgepair[0]
                    shared_edges.loc[idx, 'roi2-name'] = roi2name
                    shared_edges.loc[idx, 'roi2-edge'] = edgepair[1]

    return shared_edges

# %%===========================================================================
# Constants & prep work
# =============================================================================

import os
import girder_client
from pandas import read_csv

APIURL = 'http://candygram.neurology.emory.edu:8080/api/v1/'
SAMPLE_SLIDE_ID = '5d586d76bd4404c6b1f286ae'

gc = girder_client.GirderClient(apiUrl=APIURL)
# gc.authenticate(interactive=True)
gc.authenticate(apiKey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')

# read GTCodes dataframe
PTESTS_PATH = os.path.join(os.getcwd(), '..', '..', 'plugin_tests')
GTCODE_PATH = os.path.join(PTESTS_PATH, 'test_files', 'sample_GTcodes.csv')
GTCodes_df = read_csv(GTCODE_PATH)
GTCodes_df.index = GTCodes_df.loc[:, 'group']

# This is where masks for adjacent rois are saved
MASK_LOADPATH = os.path.join(
    PTESTS_PATH, 'test_files', 'polygon_merger_roi_masks')
maskpaths = [
    os.path.join(MASK_LOADPATH, j) for j in os.listdir(MASK_LOADPATH)
    if j.endswith('.png')]

# %%===========================================================================

verbose = True
monitorPrefix = ""

# get contours from all masks, separating edge from non-edge
# IMPORTANT -- ignore roi boundary but keep background
contkwargs = {
    'GTCodes_df': GTCodes_df,
    'get_roi_contour': False,
    'discard_nonenclosed_background': False,
    'MIN_SIZE': 2,
    'MAX_SIZE': None,
    'verbose': verbose,
}
ordinary_contours, edge_contours = get_contours_from_all_masks(
        maskpaths, contkwargs=contkwargs)

# get shared edges between rois
roiinfos = get_roi_bboxes(maskpaths)
shared_edges = _get_shared_roi_edges(roiinfos)

# %%

group = 'mostly_tumor'

# %%


def _get_merge_pairs(
        edge_contours, edgepair, group, thresh=3,
        monitorPrefix="", verbose=True):
    """Get nest dataframes and indices of which ones to merge."""
    cpr = Conditional_Print(verbose=verbose)
    _print = cpr._print

    def _get_nests_slice(ridx=1):
        Nests = edge_contours[edgepair['roi%d-name' % ridx]]
        edgevar = "touches_edge-%s" % (edgepair['roi%d-edge' % ridx])
        Nests = Nests.loc[Nests.loc[:, edgevar] == 1, :]
        Nests = Nests.loc[Nests.loc[:, 'group'] == group, :]
        return Nests

    # Nests of the same label. The nest IDs are using the index of the
    # roi dataframe from the edge_nests dictionary
    Nests1 = _get_nests_slice(1)
    Nests2 = _get_nests_slice(2)

    # to avoid redoing things, keep all polygons in a list
    polygons1 = []
    nno1 = 0
    nno1Max = Nests1.shape[0]
    for nid1, nest1 in Nests1.iterrows():
        nno1 += 1
        _print("%s: edge1 nest %d of %d" % (monitorPrefix, nno1, nno1Max))
        a
        try:
            coords = np.array(self._parse_annot_coords(nest1))
            coords[:, 0] = coords[:, 0] + roi1info['left_adj']
            coords[:, 1] = coords[:, 1] + roi1info['top_adj']
            polygons1.append((nid1, Polygon(coords)))
        except Exception as e:
            print("%s: edge1 nest %d of %d: Shapely Error (below)" % (
                    countStrBase, nno1, nno1Max))
            print(e)
    
    # go through the "other" polygons to get merge list
    to_merge = df(columns=[
        'nest1-roiname', 'nest1-nid', 'nest2-roiname', 'nest2-nid'])
    nno2 = 0
    nno2Max = Nests2.shape[0]
    for nid2, nest2 in Nests2.iterrows():
        nno2 += 1
        print("%s: edge2 nest %d of %d" % (countStrBase, nno2, nno2Max))
        try:
            coords = np.array(self._parse_annot_coords(nest2))
            coords[:, 0] = coords[:, 0] + roi2info['left_adj'] # x
            coords[:, 1] = coords[:, 1] + roi2info['top_adj'] # y
            polygon2 = Polygon(coords)
        except Exception as e:
            print("%s: edge2 nest %d of %d: Shapely Error (below)" % (
                    countStrBase, nno2, nno2Max))
            print(e)
            continue
        
        nno1Max = len(polygons1)-1
        for nno1, poly1 in enumerate(polygons1):
            print("%s: edge2 nest %d of %d: vs. edge1 nest %d of %d" % (
                    countStrBase, nno2, nno2Max, nno1, nno1Max))
            nid1, polygon1 = poly1
            if polygon1.distance(polygon2) < thresh:
                idx = to_merge.shape[0]
                to_merge.loc[idx, 'nest1-roiname'] = roi1info['roiname']
                to_merge.loc[idx, 'nest1-nid'] = nid1
                to_merge.loc[idx, 'nest2-roiname'] = roi2info['roiname']
                to_merge.loc[idx, 'nest2-nid'] = nid2
                
    return to_merge

#%%

# get merge dataframe (pairs along shared edges)
merge_df = DataFrame()
for rpno, edgepair in shared_edges.iterrows():
    edgepair = dict(edgepair)
    a
#    countStr = "%s: edge pair %d of %d" % (
#            monitorPrefix, rpno+1, shared_edges.shape[0])
#
#    # isolate nests from the two rois of interest
#    roi1info = self._get_adjusted_roiinfo(
#            edge_nests=edge_nests, roiinfos=roiinfos, 
#            roipair=roipair, roino=1, slide_info=slide_info)
#    roi2info = self._get_adjusted_roiinfo(
#            edge_nests=edge_nests, roiinfos=roiinfos, 
#            roipair=roipair, roino=2, slide_info=slide_info)
#    # get nest pairs to merge
#    to_merge = self._get_merge_pairs(
#        roi1info, roi2info, thresh=thresh, 
#        label=label, countStrBase=countStr)
#    merge_df = concat((merge_df, to_merge), axis=0)
#    
## We have every single pair to merge, including for same nest
#merge_df.reset_index(drop=True, inplace=True)








