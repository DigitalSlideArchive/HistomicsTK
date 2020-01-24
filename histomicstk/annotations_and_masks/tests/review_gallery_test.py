import copy
import unittest

import os
import girder_client
import matplotlib.pylab as plt
from pandas import read_csv
import tempfile
import shutil
import numpy as np

from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    get_bboxes_from_slide_annotations, _get_idxs_for_all_rois,
    scale_slide_annotations, get_scale_factor_and_appendStr,
    get_idxs_for_annots_overlapping_roi_by_bbox, get_image_from_htk_response,
    parse_slide_annotations_into_tables)

from histomicstk.annotations_and_masks.annotations_to_masks_handler import \
    _get_roi_bounds_by_run_mode, _visualize_annotations_on_rgb

# %%===========================================================================
# Constants & prep work

APIURL = 'http://candygram.neurology.emory.edu:8080/api/v1/'
SAMPLE_SLIDE_ID = '5d586d57bd4404c6b1f28640'

gc = girder_client.GirderClient(apiUrl=APIURL)
# gc.authenticate(interactive=True)
gc.authenticate(apiKey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')

# just a temp directory to save masks for now
# BASE_SAVEPATH = tempfile.mkdtemp()
# SAVEPATHS = {
#     'rgb': os.path.join(BASE_SAVEPATH, 'rgbs'),
#     'contours': os.path.join(BASE_SAVEPATH, 'contours'),
#     'visualization': os.path.join(BASE_SAVEPATH, 'vis'),
# }
# for _, savepath in SAVEPATHS.items():
#     os.mkdir(savepath)

# Microns-per-pixel / Magnification (either or)
MPP = 2.5  # 5.0
MAG = None

# # get annotations for slide
# slide_annotations = gc.get('/annotation/item/' + SAMPLE_SLIDE_ID)
#
# # scale up/down annotations by a factor
# sf, _ = get_scale_factor_and_appendStr(
#     gc=gc, slide_id=SAMPLE_SLIDE_ID, MPP=MPP, MAG=MAG)
# slide_annotations = scale_slide_annotations(slide_annotations, sf=sf)
#
# # get bounding box information for all annotations
# element_infos = get_bboxes_from_slide_annotations(slide_annotations)

# %%===========================================================================

# params for get_annotations_in_slide_region()
# gc=gc
slide_id = SAMPLE_SLIDE_ID
# MPP=MPP
# MAG=MAG
mode = 'manual_bounds'
bounds = {
    'XMIN': 58000, 'XMAX': 63000,
    'YMIN': 35000, 'YMAX': 39000
}
idx_for_roi = None
slide_annotations = None
element_infos = None
iou_thresh = 0.0  # set to 1.0 to only keep non-overflowing elements
linewidth = 0.2
get_rgb = True
get_visualization = True

# %%===========================================================================


def _sanity_checks_2(
        MPP, MAG, mode, bounds, idx_for_roi, get_rgb, get_visualization):

    # MPP precedes MAG
    if all([j is not None for j in (MPP, MAG)]):
        MAG = None

    # some sanity checks

    for mf in (MPP, MAG):
        if mf is not None:
            assert mf > 0, "MPP or MAG must be positive."

    if mode in ['wsi', 'min_bounding_box']:
        bounds = None
        idx_for_roi = None

    if idx_for_roi is not None:
        mode = 'polygonal_bounds'
    elif bounds is not None:
        mode = 'manual_bounds'

    assert mode in [
        'wsi', 'min_bounding_box', 'manual_bounds', 'polygonal_bounds'], \
        "mode %s not recognized" % mode

    if get_visualization:
        assert get_rgb, "cannot get visualization without rgb."

    return MPP, MAG, mode, bounds, idx_for_roi, get_rgb, get_visualization

# %%===========================================================================


MPP, MAG, mode, bounds, idx_for_roi, get_rgb, get_visualization = \
    _sanity_checks_2(
        MPP, MAG, mode, bounds, idx_for_roi, get_rgb, get_visualization)

# %%===========================================================================

# calculate the scale factor
sf, appendStr = get_scale_factor_and_appendStr(
    gc=gc, slide_id=slide_id, MPP=MPP, MAG=MAG)

if slide_annotations is not None:
    assert element_infos is not None, "must also provide element_infos"
else:
    # get annotations for slide
    slide_annotations = gc.get('/annotation/item/' + slide_id)

    # scale up/down annotations by a factor
    slide_annotations = scale_slide_annotations(slide_annotations, sf=sf)

    # get bounding box information for all annotations -> scaled by sf
    element_infos = get_bboxes_from_slide_annotations(slide_annotations)

# Detemine get region based on run mode, keeping in mind that it
# must be at BASE MAGNIFICATION coordinates before it is passed
# on to get_mask_from_slide()
bounds = _get_roi_bounds_by_run_mode(
    gc=gc, slide_id=slide_id, mode=mode, bounds=bounds,
    element_infos=element_infos, idx_for_roi=idx_for_roi, sf=sf)
result = {'bounds': bounds, }

# %%===========================================================================


def _keep_relevant_elements_for_roi(
        element_infos, mode='manual_bounds',
        idx_for_roi=None, iou_thresh=0.0, roiinfo=None):

    # This stores information about the ROI like bounds, slide_name, etc
    # Allows passing many parameters and good forward/backward compatibility
    if roiinfo is None:
        roiinfo = dict()

    if mode != "polygonal_bounds":
        # add to bounding boxes dataframe
        element_infos = element_infos.append(
            {'xmin': int(roiinfo['XMIN'] * sf),
             'xmax': int(roiinfo['XMAX'] * sf),
             'ymin': int(roiinfo['YMIN'] * sf),
             'ymax': int(roiinfo['YMAX'] * sf)},
            ignore_index=True)
        idx_for_roi = element_infos.shape[0] - 1

    # isolate annotations that potentially overlap roi
    overlaps = get_idxs_for_annots_overlapping_roi_by_bbox(
        element_infos, idx_for_roi=idx_for_roi, iou_thresh=iou_thresh)
    elinfos_roi = element_infos.loc[overlaps, :]

    # update roiinfo -- remember, annotation elements can be
    # really large and extend beyond the bounds asked by the user.
    # since we're not parsing the polygons into mask form here, and
    # therefore we're not 'cropping' the polygons to the requested bounds,
    # we extend the requested bounds themselves to accomodate the overflowing
    # annotations. Alternatively, if this behavior is not desired, the user
    # can pass iou_thresh = 1.0 and only keep annotations that are fully
    # enclosed within the requested bounds
    roiinfo['XMIN'] = int(np.min(elinfos_roi.xmin))
    roiinfo['YMIN'] = int(np.min(elinfos_roi.ymin))
    roiinfo['XMAX'] = int(np.max(elinfos_roi.xmax))
    roiinfo['YMAX'] = int(np.max(elinfos_roi.ymax))
    roiinfo['BBOX_WIDTH'] = roiinfo['XMAX'] - roiinfo['XMIN']
    roiinfo['BBOX_HEIGHT'] = roiinfo['YMAX'] - roiinfo['YMIN']

    # scale back coords
    roiinfo = {k: int(v / sf) for k, v in roiinfo.items()}

    return elinfos_roi, roiinfo


# %%===========================================================================

# only keep relevale elements and get updated bounds
elinfos_roi, bounds = _keep_relevant_elements_for_roi(
    element_infos, mode=mode, idx_for_roi=idx_for_roi,
    iou_thresh=iou_thresh, roiinfo=copy.deepcopy(bounds))

# %%===========================================================================

# get RGB
if get_rgb:
    getStr = \
        "/item/%s/tiles/region?left=%d&right=%d&top=%d&bottom=%d" \
        % (slide_id,
           bounds['XMIN'], bounds['XMAX'],
           bounds['YMIN'], bounds['YMAX'])
    getStr += appendStr
    resp = gc.get(getStr, jsonResp=False)
    rgb = get_image_from_htk_response(resp)
    result['rgb'] = rgb


# %%===========================================================================


def _trim_slide_annotations_to_roi(slide_annotations, elinfos_roi):

    annotations = copy.deepcopy(slide_annotations)

    # unique relevent annotation document indices
    unique_annidxs = np.int32(np.unique(elinfos_roi.loc[:, "annidx"]))

    # unique relevant element indixes in each annotation document
    unique_elementidxs = []
    for annidx in unique_annidxs:
        eleidxs = elinfos_roi.loc[
            elinfos_roi.loc[:, 'annidx'] == annidx, 'elementidx']
        unique_elementidxs.append(np.int32(np.unique(eleidxs)))

    # now slice as needed
    annotations_slice = np.array(annotations)[unique_annidxs].tolist()
    for annidx in range(len(annotations_slice)):
        elements_original = annotations_slice[annidx]['annotation']['elements']
        annotations_slice[annidx]['annotation']['elements'] = np.array(
            elements_original)[unique_elementidxs[annidx]].tolist()

    return annotations_slice


# %%===========================================================================

# find relevant portion from slide annotations to use
annotations_slice = _trim_slide_annotations_to_roi(
    slide_annotations, elinfos_roi=elinfos_roi)

# tabularize to use contours
_, contours_list = parse_slide_annotations_into_tables(
    annotations_slice, x_offset=int(bounds['XMIN'] * sf),
    y_offset=int(bounds['YMIN'] * sf),)
contours_list = contours_list.to_dict(orient='records')
result['contours'] = contours_list

# get visualization of annotations on RGB
if get_visualization:
    result['visualization'] = _visualize_annotations_on_rgb(
        rgb=rgb, contours_list=contours_list, linewidth=linewidth)

# %%===========================================================================
