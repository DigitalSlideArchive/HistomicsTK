# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 18:33:48 2019.

@author: tageldim

"""

import os
import numpy as np
from pandas import read_csv
from imageio import imwrite
from shapely.geometry.polygon import Polygon

from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    # from annotation_and_mask_utils import (
    get_bboxes_from_slide_annotations, _get_idxs_for_all_rois,
    get_idxs_for_annots_overlapping_roi_by_bbox, _get_element_mask,
    _get_and_add_element_to_roi)

# %% =====================================================================


def get_roi_mask(
        slide_annotations, element_infos, GTCodes_df,
        idx_for_roi, iou_thresh=0.0, roiinfo=None,
        crop_to_roi=True, use_shapely=True,
        verbose=False, monitorPrefix=""):
    """Parse annotations and gets a ground truth mask for a single ROI.

    This will look at all slide annotations and get ones that
    overlap with the region of interest (ROI) and assigns them to mask.

    Parameters
    -----------
    slide_annotations : list of dicts
        response from server request
    element_infos : pandas DataFrame.
        The columns annidx and elementidx
        encode the dict index of annotation document and element,
        respectively, in the original slide_annotations list of dictionaries.
        This can be obain by get_bboxes_from_slide_annotations() method
    GTCodes_df : pandas Dataframe
        the ground truth codes and information dataframe.
        WARNING: Modified indide this method so pass a copy.
        This is a dataframe that is indexed by the annotation group name and
        has the following columns:
        - group: group name of annotation (string), eg. mostly_tumor
        - overlay_order: int, how early to place the annotation in the
        mask. Larger values means this annotation group is overlayed
        last and overwrites whatever overlaps it.
        - GT_code: int, desired ground truth code (in the mask)
        Pixels of this value belong to corresponding group (class)
        - is_roi: Flag for whether this group encodes an ROI
        - is_background_class: Flag, whether this group is the default
        fill value inside the ROI. For example, you may descide that
        any pixel inside the ROI is considered stroma.
    idx_for_roi : int
        index of ROI within the element_infos dataframe.
    iou_thresh : float
        how much bounding box overlap is enough to
        consider an annotation to belong to the region of interest
    roiinfo : pandas series or dict
        contains information about the roi. Keys will be added to this
        index containing info about the roi like bounding box
        location and size.
    crop_to_roi : bool
        flag of whether to crop polygons to roi
        (prevent overflow beyond roi edge)
    use_shapely : bool
        flag of whether to precisely determine whether an element
        belongs to an ROI using shapely polygons. Slightly slower. If
        set to False, overlapping bounding box is used as a cheap but
        less precise indicator of inclusion.
    verbose : bool
        Print progress to screen?
    monitorPrefix : str
        text to prepend to printed statements

    Returns
    --------
    Np array
        (N x 2), where pixel values encode class membership.
        IMPORTANT NOTE: Zero pixels have special meaning and do NOT
        encode specific ground truth class. Instead, they simply
        mean Outside ROI and should be IGNORED during model training
        or evaluation.
    Dict
        information about ROI

    """
    # This stores information about the ROI like bounds, slide_name, etc
    # Allows passing many parameters and good forward/backward compatibility
    if roiinfo is None:
        roiinfo = dict()

    # isolate annotations that potentially overlap (belong to) mask (incl. ROI)
    overlaps = get_idxs_for_annots_overlapping_roi_by_bbox(
        element_infos, idx_for_roi=idx_for_roi, iou_thresh=iou_thresh)
    idxs_for_all_rois = _get_idxs_for_all_rois(
        GTCodes=GTCodes_df, element_infos=element_infos)
    overlaps = list(set(overlaps) - set(idxs_for_all_rois))
    elinfos_roi = element_infos.loc[[idx_for_roi, ] + overlaps, :]

    # Add roiinfo
    roiinfo['XMIN'] = int(np.min(elinfos_roi.xmin))
    roiinfo['YMIN'] = int(np.min(elinfos_roi.ymin))
    roiinfo['XMAX'] = int(np.max(elinfos_roi.xmax))
    roiinfo['YMAX'] = int(np.max(elinfos_roi.ymax))
    roiinfo['BBOX_WIDTH'] = roiinfo['XMAX'] - roiinfo['XMIN']
    roiinfo['BBOX_HEIGHT'] = roiinfo['YMAX'] - roiinfo['YMIN']

    # get roi polygon
    if use_shapely:
        coords, _ = _get_element_mask(
            elinfo=elinfos_roi.loc[idx_for_roi],
            slide_annotations=slide_annotations)
        roi_polygon = Polygon(coords)

    # Init mask
    ROI = np.zeros(
        (roiinfo['BBOX_HEIGHT'], roiinfo['BBOX_WIDTH']), dtype=np.uint8)

    # only parse if roi is polygonal or rectangular
    if elinfos_roi.loc[idx_for_roi, 'type'] == 'point':
        raise Exception("roi cannot be a point!")

    # make sure ROI is overlayed first & assigned background class if relevant
    roi_group = elinfos_roi.loc[idx_for_roi, 'group']
    GTCodes_df.loc[roi_group, 'overlay_order'] = np.min(
        GTCodes_df.loc[:, 'overlay_order']) - 1
    bck_classes = GTCodes_df.loc[
        GTCodes_df.loc[:, 'is_background_class'] == 1, :]
    if bck_classes.shape[0] > 0:
        GTCodes_df.loc[
            roi_group, 'GT_code'] = bck_classes.iloc[0, :]['GT_code']

    # Add annotations in overlay order
    overlay_orders = sorted(set(GTCodes_df.loc[:, 'overlay_order']))
    N_elements = elinfos_roi.shape[0]
    elNo = 0
    for overlay_level in overlay_orders:

        # get indices of relevant groups
        relevant_groups = list(GTCodes_df.loc[
            GTCodes_df.loc[:, 'overlay_order'] == overlay_level, 'group'])
        relIdxs = []
        for group_name in relevant_groups:
            relIdxs.extend(list(elinfos_roi.loc[
                elinfos_roi.group == group_name, :].index))

        # get relevnt infos and sort from largest to smallest (by bbox area)
        # so that the smaller elements are layered last. This helps partially
        # address issues describe in:
        # https://github.com/DigitalSlideArchive/HistomicsTK/issues/675
        elinfos_relevant = elinfos_roi.loc[relIdxs, :].copy()
        elinfos_relevant.sort_values(
            'bbox_area', axis=0, ascending=False, inplace=True)

        # Go through elements and add to ROI mask
        for elId, elinfo in elinfos_relevant.iterrows():

            elNo += 1
            elcountStr = "%s: Overlay level %d: Element %d of %d: %s" % (
                monitorPrefix, overlay_level, elNo, N_elements, elinfo['group'])
            if verbose:
                print(elcountStr)

            # now add element to ROI
            ROI = _get_and_add_element_to_roi(
                elinfo=elinfo, slide_annotations=slide_annotations, ROI=ROI,
                roiinfo=roiinfo, roi_polygon=roi_polygon,
                GT_code=GTCodes_df.loc[elinfo['group'], 'GT_code'],
                use_shapely=use_shapely, verbose=verbose,
                monitorPrefix=elcountStr)

            # save a copy of ROI-only mask to crop to it later if needed
            if crop_to_roi and (overlay_level == GTCodes_df.loc[
                    roi_group, 'overlay_order']):
                roi_only_mask = ROI.copy()

    # Crop polygons to roi if needed (prevent 'overflow' beyond roi edge)
    if crop_to_roi:
        ROI[roi_only_mask == 0] = 0

    # tighten boundary --remember, so far we've use element bboxes to
    # make an over-estimated margin around ROI boundary.
    nz = np.nonzero(ROI)
    ymin, xmin = [np.min(arr) for arr in nz]
    ymax, xmax = [np.max(arr) for arr in nz]
    ROI = ROI[ymin:ymax, xmin:xmax]

    # update roi offset
    roiinfo['XMIN'] += xmin
    roiinfo['YMIN'] += ymin
    roiinfo['XMAX'] += xmin
    roiinfo['YMAX'] += ymin
    roiinfo['BBOX_WIDTH'] = roiinfo['XMAX'] - roiinfo['XMIN']
    roiinfo['BBOX_HEIGHT'] = roiinfo['YMAX'] - roiinfo['YMIN']

    return ROI, roiinfo

# %% =====================================================================


def get_all_roi_masks_for_slide(
        gc, slide_id, GTCODE_PATH, MASK_SAVEPATH, slide_name=None,
        verbose=True, monitorPrefix="", get_roi_mask_kwargs=dict()):
    """Parse annotations and saves ground truth masks for ALL ROIs.

    Get all ROIs in a single slide. This is a wrapper around get_roi_mask()
    which should be referred to for implementation details.

    Parameters
    -----------
    gc : object
        girder client object to make requests, for example:
        gc = girder_client.GirderClient(apiUrl = APIURL)
        gc.authenticate(interactive=True)
    slide_id : str
        girder id for item (slide)
    GTCODE_PATH : str
        path to the ground truth codes and information
        csv file. Refer to the docstring of get_roi_mask() for more info.
    MASK_SAVEPATH : str
        path to directory to save ROI masks
    slide_name (optional) : str
        If not given, it's inferred using a server request using girder client.
    verbose (optional) : bool
        Print progress to screen?
    monitorPrefix (optional) : str
        text to prepend to printed statements
    get_roi_mask_kwargs : dict
        extra kwargs for get_roi_mask()

    Returns
    --------
    list of strs
        save paths for ROIs

    """
    # if not given, assign name of first file associated with item
    if slide_name is None:
        resp = gc.get('/item/%s/files' % slide_id)
        slide_name = resp[0]['name']
        slide_name = slide_name[:slide_name.rfind('.')]

    # read ground truth codes and information
    GTCodes = read_csv(GTCODE_PATH)
    GTCodes.index = GTCodes.loc[:, 'group']
    if any(GTCodes.loc[:, 'GT_code'] <= 0):
        raise Exception("All GT_code must be > 0")

    # get annotations for slide
    slide_annotations = gc.get('/annotation/item/' + slide_id)

    # get bounding box information for all annotations
    element_infos = get_bboxes_from_slide_annotations(slide_annotations)

    # get indices of rois
    idxs_for_all_rois = _get_idxs_for_all_rois(
        GTCodes=GTCodes, element_infos=element_infos)

    savenames = []

    for roino, idx_for_roi in enumerate(idxs_for_all_rois):

        roicountStr = "%s: roi %d of %d" % (
            monitorPrefix, roino + 1, len(idxs_for_all_rois))

        # get roi mask and info
        ROI, roiinfo = get_roi_mask(
            slide_annotations=slide_annotations, element_infos=element_infos,
            GTCodes_df=GTCodes.copy(), idx_for_roi=idx_for_roi,
            monitorPrefix=roicountStr, **get_roi_mask_kwargs)

        # now save roi
        ROINAMESTR = "%s_left-%d_top-%d_mag-BASE" % (
            slide_name, roiinfo['XMIN'], roiinfo['YMIN'])
        savename = os.path.join(MASK_SAVEPATH, ROINAMESTR + ".png")
        if verbose:
            print("%s: Saving %s\n" % (roicountStr, savename))
        imwrite(im=ROI, uri=savename)

        savenames.append(savename)

    return savenames

# %% =====================================================================
