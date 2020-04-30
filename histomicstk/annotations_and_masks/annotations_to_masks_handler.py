# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 18:33:48 2019.

@author: tageldim

"""

import os
import numpy as np
from pandas import DataFrame
from imageio import imwrite
from shapely.geometry.polygon import Polygon
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass
from matplotlib.patches import Polygon as mpPolygon
import io
from PIL import Image
import copy

from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    get_bboxes_from_slide_annotations, _get_idxs_for_all_rois,
    get_idxs_for_annots_overlapping_roi_by_bbox, _get_element_mask,
    _get_and_add_element_to_roi, scale_slide_annotations,
    get_image_from_htk_response, get_scale_factor_and_appendStr)
from histomicstk.annotations_and_masks.masks_to_annotations_handler import (
    get_contours_from_mask)

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
        WARNING: Modified inside this method so pass a copy.
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
                monitorPrefix, overlay_level, elNo, N_elements,
                elinfo['group'])
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


def get_mask_from_slide(
        GTCodes_dict, roiinfo, slide_annotations,
        element_infos, sf=1.0, get_roi_mask_kwargs=None):
    """Parse region from the slide and get its corresponding labeled mask.

    This is a wrapper around get_roi_mask() which should be referred to for
    implementation details. If roiinfo is None, all annotations in the slide
    are parsed into labeled image (mask) form. Otherwise, the bounding box
    coordinates in roiinfo are used.

    Parameters
    -----------
    GTCodes_dict : dict
        the ground truth codes and information dict.
        This is a dict that is indexed by the annotation group name and
        each entry is in turn a dict with the following keys:
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

    roiinfo : dict or None
        if not None, has keys 'XMIN', 'XMAX', 'YMIN', 'YMAX' for slide
        region coordinates (AT BASE MAGNIFICATION) to get labeled image
        (mask) for.

    sf : float
        scale factor to multiple coordinates (eg 0.5 would halve size)

    slide_annotations : list
        Make sure you have used
        scale_slide_annotations() to scale them up/down by sf BEFOREHAND.

    element_infos : pandas DataFrame.
        The columns annidx and elementidx
        encode the dict index of annotation document and element,
        respectively, in the original slide_annotations list of dictionaries.
        This can be obained by get_bboxes_from_slide_annotations() method.
        Make sure you have used scale_slide_annotations().

    get_roi_mask_kwargs : dict
        extra kwargs for get_roi_mask()

    Returns
    --------
    Np array
        (N x 2), where pixel values encode class membership.
        IMPORTANT NOTE: Zero pixels have special meaning and do NOT
        encode specific ground truth class. Instead, they simply
        mean Outside mask and should be IGNORED during model training
        or evaluation.

    Dict
        information about mask

    """
    # convert from dict to required dataframe
    if get_roi_mask_kwargs is None:
        get_roi_mask_kwargs = dict()
    GTCodes = DataFrame.from_dict(GTCodes_dict, orient='index')

    # some sanity checks
    assert all([j in GTCodes.columns for j in [
        'group', 'overlay_order', 'GT_code', 'is_roi', 'is_background_class',
        'color']]), "GTCodes_dict does not follow schema"
    assert all(GTCodes.loc[:, 'GT_code'] > 0), "All GT_code must be > 0"
    assert sf > 0, "sf must be positive."
    assert (roiinfo['XMAX'] > roiinfo['XMIN']) and (
        roiinfo['YMAX'] > roiinfo['YMIN'])

    # use given ROI bounds, after scaling
    XMIN = int(roiinfo['XMIN'] * sf)
    YMIN = int(roiinfo['YMIN'] * sf)
    XMAX = int(roiinfo['XMAX'] * sf)
    YMAX = int(roiinfo['YMAX'] * sf)
    WIDTH = XMAX - XMIN
    HEIGHT = YMAX - YMIN

    # add to slide annotations list
    slide_annotations.append({'annotation': {
        'description': '',
        'elements': [
            {'center': [int(XMIN + WIDTH / 2), int(YMIN + HEIGHT / 2), 0],
             'width': WIDTH,
             'height': HEIGHT,
             'normal': [0, 0, 1],
             'rotation': 0,
             'group': 'super_roi',
             'label': {'value': 'super_roi'},
             'lineColor': 'rgb(0, 0, 0)',
             'fillColor': 'rgba(0, 0, 0, 0)',
             'lineWidth': 4.6,
             'type': 'rectangle', }
        ],
        'name': 'superROI'},
    })

    # add to bounding boxes dataframe
    element_infos = element_infos.append({
        'annidx': len(slide_annotations) - 1,
        'elementidx': 0,
        'type': 'rectangle',
        'group': 'super_roi',
        'xmin': XMIN,
        'xmax': XMAX,
        'ymin': YMIN,
        'ymax': YMAX,
        'bbox_area': WIDTH * HEIGHT,
    }, ignore_index=True)

    # find roi and background codes to use later
    roi_codes = list(GTCodes.loc[GTCodes.loc[:, 'is_roi'] == 1, "GT_code"])
    bck_code = GTCodes.loc[
        GTCodes.loc[:, 'is_background_class'] == 1, "GT_code"]
    if bck_code.shape[0] > 0:
        bck_code = int(bck_code.iloc[0])
    else:
        bck_code = 0

    # add to gtcodes dataframe
    assert np.max(GTCodes.loc[:, 'GT_code']) < 255
    GTCodes.loc[:, 'is_roi'] = 0  # treat other ROIs as ordinary annotations
    GTCodes.loc[:, 'is_background_class'] = 0  # we'll adjust later
    GTCodes = GTCodes.append({
        'GT_code': 255,
        'overlay_order': 0,
        'color': 'rgb(0,0,0)',
        'group': 'super_roi',
        'is_background_class': 0,
        'is_roi': 1,
    }, ignore_index=True)
    GTCodes.index = GTCodes.loc[:, 'group']

    # now get mask
    ROI, roiinfo = get_roi_mask(
        slide_annotations=slide_annotations, element_infos=element_infos,
        GTCodes_df=GTCodes.copy(),
        idx_for_roi=element_infos.index[-1],  # <- bounding roi
        **get_roi_mask_kwargs)
    ROI[ROI == 255] = 0

    # replace roi codes with background code
    for roi_code in roi_codes:
        ROI[ROI == roi_code] = bck_code

    # scale back coords
    roiinfo = {k: int(v / sf) for k, v in roiinfo.items()}

    return ROI, roiinfo

# %% =====================================================================


def _visualize_annotations_on_rgb(
        rgb, contours_list, linewidth=0.2, x_offset=0, y_offset=0,
        text=False):

    # later on flipped by matplotlib for weird reason
    rgb = np.flipud(rgb)

    fig = plt.figure(
        figsize=(rgb.shape[1] / 1000, rgb.shape[0] / 1000), dpi=100)
    ax = plt.subplot(111)
    ax.imshow(rgb)

    plt.axis('off')
    ax = plt.gca()
    ax.set_xlim(0.0, rgb.shape[1])
    ax.set_ylim(0.0, rgb.shape[0])

    for idx, ann in enumerate(contours_list):
        xy = np.array([
            [int(j) for j in ann[k].split(",")]
            for k in ('coords_x', 'coords_y')]).T
        xy[:, 0] = xy[:, 0] - x_offset
        xy[:, 1] = rgb.shape[0] - (xy[:, 1] - y_offset) + 1
        polygon = mpPolygon(
            xy=xy,
            color=[int(j) / 255 for j in ann['color'].split(
                'rgb(')[1][:-1].split(',')],
            closed=True, fill=False,
            linewidth=linewidth,
        )
        ax.add_patch(polygon)

        # add label text
        if text:
            txtshift = 0
            size = 1e-4 * rgb.shape[1]
            ax.text(
                int(np.min(xy[:, 0])),
                int(np.max(xy[:, 1])) - txtshift,
                ann['group'][:5],
                color='w', fontsize=size, backgroundcolor="none",
            )

    ax.axis('off')
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', pad_inches=0, dpi=1000)
    buf.seek(0)
    rgb_vis = np.uint8(Image.open(buf))[..., :3]
    plt.close()

    return rgb_vis

# %% =====================================================================


def _sanity_checks(
        MPP, MAG, mode, bounds, idx_for_roi, get_roi_mask_kwargs,
        get_rgb, get_contours, get_visualization):

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
        get_contours = True
        assert get_rgb, "cannot get visualization without rgb."

    if not get_roi_mask_kwargs['crop_to_roi']:
        assert (not get_rgb) and (not get_visualization), \
            "Handling overflowing annotations while also getting RGB is" \
            "not currently supported."

    return (
        MPP, MAG, mode, bounds, idx_for_roi, get_roi_mask_kwargs,
        get_rgb, get_contours, get_visualization)


def _get_roi_bounds_by_run_mode(
        gc, slide_id, mode, bounds, element_infos, idx_for_roi, sf):

    if mode == 'polygonal_bounds':
        # get bounds based on specified polygonal/rotated roi
        elinfo = element_infos.loc[idx_for_roi]
        bounds = {
            'XMIN': int(elinfo['xmin'] / sf),
            'XMAX': int(elinfo['xmax'] / sf),
            'YMIN': int(elinfo['ymin'] / sf),
            'YMAX': int(elinfo['ymax'] / sf),
        }

    elif mode == 'manual_bounds':
        assert (bounds['XMAX'] > bounds['XMIN']) and (
            bounds['YMAX'] > bounds['YMIN'])

    elif mode == 'min_bounding_box':
        # get minimum box for all annotations in slide
        bounds = {
            'XMIN': int(np.min(element_infos.xmin) / sf),
            'YMIN': int(np.min(element_infos.ymin) / sf),
            'XMAX': int(np.max(element_infos.xmax) / sf),
            'YMAX': int(np.max(element_infos.ymax) / sf),
        }
    else:
        # get scaled up/down version of mask of whole slide
        slide_info = gc.get("/item/%s/tiles" % slide_id)
        bounds = {
            'XMIN': 0,
            'XMAX': slide_info['sizeX'],
            'YMIN': 0,
            'YMAX': slide_info['sizeY'],
        }

    return bounds


def _get_rgb_and_pad_roi(gc, slide_id, bounds, appendStr, ROI):

    getStr = \
        "/item/%s/tiles/region?left=%d&right=%d&top=%d&bottom=%d" \
        % (slide_id,
           bounds['XMIN'], bounds['XMAX'],
           bounds['YMIN'], bounds['YMAX'])
    getStr += appendStr
    resp = gc.get(getStr, jsonResp=False)
    rgb = get_image_from_htk_response(resp)

    # sometimes there's a couple of pixel difference d.t. rounding, so pad
    pad_y = rgb.shape[0] - ROI.shape[0]
    pad_x = rgb.shape[1] - ROI.shape[1]
    assert all([np.abs(j) < 4 for j in (pad_y, pad_x)]), \
        "too much difference in size between image and mask."\
        "something is wrong!"

    if pad_y > 0:
        ROI = np.pad(ROI, pad_width=((0, pad_y), (0, 0)), mode='constant')
    elif pad_y < 0:
        ROI = ROI[:pad_y, :]

    if pad_x > 0:
        ROI = np.pad(ROI, pad_width=((0, 0), (0, pad_x)), mode='constant')
    elif pad_x < 0:
        ROI = ROI[:, :pad_x]

    return rgb, ROI


def get_image_and_mask_from_slide(
        gc, slide_id, GTCodes_dict,
        MPP=5.0, MAG=None, mode='min_bounding_box',
        bounds=None, idx_for_roi=None,
        slide_annotations=None, element_infos=None,
        get_roi_mask_kwargs=None, get_contours_kwargs=None, linewidth=0.2,
        get_rgb=True, get_contours=True, get_visualization=True):
    """Parse region from the slide and get its corresponding labeled mask.

    This is a wrapper around get_roi_mask() which should be referred to for
    implementation details.

    Parameters
    -----------
    gc : object
        girder client object to make requests, for example:
        gc = girder_client.GirderClient(apiUrl = APIURL)
        gc.authenticate(interactive=True)

    slide_id : str
        girder id for item (slide)

    GTCodes_dict : dict
        the ground truth codes and information dict.
        This is a dict that is indexed by the annotation group name and
        each entry is in turn a dict with the following keys:
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

    MPP : float or None
        Microns-per-pixel -- best use this as it's more well-defined than
        magnification which is more scanner/manufacturer specific.
        MPP of 0.25 often roughly translates to 40x

    MAG : float or None
        If you prefer to use whatever magnification is reported in slide.
        If neither MPP or MAG is provided, everything is retrieved without
        scaling at base (scan) magnification.

    mode : str
        This specifies which part of the slide to get the mask from. Allowed
        modes include the following
        - wsi: get scaled up/down version of mask of whole slide
        - min_bounding_box: get minimum box for all annotations in slide
        - manual_bounds: use given ROI bounds provided by the 'bounds' param
        - polygonal_bounds: use the idx_for_roi param to get coordinates

    bounds : dict or None
        if not None, has keys 'XMIN', 'XMAX', 'YMIN', 'YMAX' for slide
        region coordinates (AT BASE MAGNIFICATION) to get labeled image
        (mask) for. Use this with the 'manual_bounds' run mode.

    idx_for_roi : int
        index of ROI within the element_infos dataframe.
        Use this with the 'polygonal_bounds' run mode.

    slide_annotations : list or None
        Give this parameter to avoid re-getting slide annotations. If you do
        provide the annotations, though, make sure you have used
        scale_slide_annotations() to scale them up/down by sf BEFOREHAND.

    element_infos : pandas DataFrame.
        The columns annidx and elementidx
        encode the dict index of annotation document and element,
        respectively, in the original slide_annotations list of dictionaries.
        This can be obained by get_bboxes_from_slide_annotations() method.
        Make sure you have used scale_slide_annotations().

    get_roi_mask_kwargs : dict
        extra kwargs for get_roi_mask()

    get_contours_kwargs : dict
        extra kwargs for get_contours_from_mask()

    linewidth : float
        visualization line width

    get_rgb: bool
        get rgb image?

    get_contours : bool
        get annotation contours? (relative to final mask)

    get_visualization : bool
        get overlayed annotation bounds over RGB for visualization

    Returns
    --------
    dict
        Results dict containing one or more of the following keys
        bounds: dict of bounds at scan magnification
        ROI - (mxn) labeled image (mask)
        rgb - (mxnx3 np array) corresponding rgb image
        contours - list, each entry is a dict version of a row from the output
        of masks_to_annotations_handler.get_contours_from_mask()
        visualization - (mxnx3 np array) visualization overlay

    """
    get_roi_mask_kwargs = get_roi_mask_kwargs or {}
    get_contours_kwargs = get_contours_kwargs or {}
    # important sanity checks
    (MPP, MAG, mode, bounds, idx_for_roi, get_roi_mask_kwargs,
     get_rgb, get_contours, get_visualization) = _sanity_checks(
        MPP, MAG, mode, bounds, idx_for_roi, get_roi_mask_kwargs,
        get_rgb, get_contours, get_visualization)

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

    # get mask for specified area
    if mode == 'polygonal_bounds':
        # get roi mask and info
        ROI, _ = get_roi_mask(
            slide_annotations=slide_annotations, element_infos=element_infos,
            GTCodes_df=DataFrame.from_dict(GTCodes_dict, orient='index'),
            idx_for_roi=idx_for_roi, **get_roi_mask_kwargs)
    else:
        ROI, _ = get_mask_from_slide(
            GTCodes_dict=GTCodes_dict, roiinfo=copy.deepcopy(bounds),
            slide_annotations=slide_annotations, element_infos=element_infos,
            sf=sf, get_roi_mask_kwargs=get_roi_mask_kwargs)

    # get RGB
    if get_rgb:
        rgb, ROI = _get_rgb_and_pad_roi(
            gc=gc, slide_id=slide_id, bounds=bounds,
            appendStr=appendStr, ROI=ROI)
        result['rgb'] = rgb

    # pack result (we have to do it here in case of padding)
    result['ROI'] = ROI

    # get contours
    if get_contours:
        contours_list = get_contours_from_mask(
            MASK=ROI,
            GTCodes_df=DataFrame.from_dict(GTCodes_dict, orient='index'),
            **get_contours_kwargs)
        contours_list = contours_list.to_dict(orient='records')
        result['contours'] = contours_list

    # get visualization of annotations on RGB
    if get_visualization:
        result['visualization'] = _visualize_annotations_on_rgb(
            rgb=rgb, contours_list=contours_list, linewidth=linewidth)

    return result

# %% =====================================================================


def get_all_rois_from_slide(
        gc, slide_id, GTCodes_dict, save_directories,
        get_image_and_mask_from_slide_kwargs=None,
        slide_name=None, verbose=True, monitorPrefix="", ):
    """Parse annotations and saves ground truth masks for ALL ROIs.

    Get all ROIs in a single slide. This is mainly uses
    get_image_and_mask_from_slide(), which should be referred to
    for implementation details.

    Parameters
    -----------
    gc : object
        girder client object to make requests, for example:
        gc = girder_client.GirderClient(apiUrl = APIURL)
        gc.authenticate(interactive=True)

    slide_id : str
        girder id for item (slide)

    GTCodes_dict : dict
        the ground truth codes and information dict.
        This is a dict that is indexed by the annotation group name and
        each entry is in turn a dict with the following keys:
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

    save_directories : dict
        paths to directories to save data. Each entry is a string, and the
        following keys are allowed
        - ROI: path to save masks (labeled images)
        - rgb: path to save rgb images
        - contours: path to save annotation contours
        - visualization: path to save rgb visualzation overlays

    get_image_and_mask_from_slide_kwargs : dict
        kwargs to pass to get_image_and_mask_from_slide()
        default values are assigned if speceific parameters are not given.

    slide_name : str or None
        If not given, it's inferred using a server request using girder client.

    verbose : bool
        Print progress to screen?

    monitorPrefix : str
        text to prepend to printed statements

    Returns
    --------
    list of dicts
        each entry contains the following keys
        - ROI: path to saved mask (labeled image)
        - rgb: path to saved rgb image
        - contours: path to saved annotation contours
        - visualization: path to saved rgb visualzation overlay

    """
    # assign defaults if nothing given
    default_keyvalues = {
        'MPP': 5.0,
        'MAG': None,
        'get_roi_mask_kwargs': {
            'iou_thresh': 0.0, 'crop_to_roi': True,
            'use_shapely': True, 'verbose': False},
        'get_contours_kwargs': {
            'groups_to_get': None,
            'roi_group': 'roi',
            'get_roi_contour': True,
            'discard_nonenclosed_background': True,
            'background_group': 'mostly_stroma',
            'MIN_SIZE': 10, 'MAX_SIZE': None,
            'verbose': False, 'monitorPrefix': ""
        },
        'get_rgb': True,
        'get_contours': True,
        'get_visualization': True,
    }

    kvp = get_image_and_mask_from_slide_kwargs or {}  # for easy referencing
    for k, v in default_keyvalues.items():
        if k not in kvp.keys():
            kvp[k] = v

    # convert to df and sanity check
    GTCodes_df = DataFrame.from_dict(GTCodes_dict, orient='index')
    if any(GTCodes_df.loc[:, 'GT_code'] <= 0):
        raise Exception("All GT_code must be > 0")

    # if not given, assign name of first file associated with girder item
    if slide_name is None:
        resp = gc.get('/item/%s/files' % slide_id)
        slide_name = resp[0]['name']
        slide_name = slide_name[:slide_name.rfind('.')]

    # get annotations for slide
    slide_annotations = gc.get('/annotation/item/' + slide_id)

    # scale up/down annotations by a factor
    sf, _ = get_scale_factor_and_appendStr(
        gc=gc, slide_id=slide_id, MPP=kvp['MPP'], MAG=kvp['MAG'])
    slide_annotations = scale_slide_annotations(slide_annotations, sf=sf)

    # get bounding box information for all annotations
    element_infos = get_bboxes_from_slide_annotations(slide_annotations)

    # get idx of all 'special' roi annotations
    idxs_for_all_rois = _get_idxs_for_all_rois(
        GTCodes=GTCodes_df, element_infos=element_infos)

    savenames = []

    for roino, idx_for_roi in enumerate(idxs_for_all_rois):

        roicountStr = "%s: roi %d of %d" % (
            monitorPrefix, roino + 1, len(idxs_for_all_rois))

        # get specified area
        roi_out = get_image_and_mask_from_slide(
            gc=gc, slide_id=slide_id, GTCodes_dict=GTCodes_dict,
            mode='polygonal_bounds', idx_for_roi=idx_for_roi,
            slide_annotations=slide_annotations,
            element_infos=element_infos, **kvp)

        # now save roi (mask, rgb, contours, vis)

        this_roi_savenames = dict()
        ROINAMESTR = "%s_left-%d_top-%d_bottom-%d_right-%d" % (
            slide_name,
            roi_out['bounds']['XMIN'], roi_out['bounds']['YMIN'],
            roi_out['bounds']['YMAX'], roi_out['bounds']['XMAX'])

        for imtype in ['ROI', 'rgb', 'visualization']:
            if imtype in roi_out.keys():
                savename = os.path.join(
                    save_directories[imtype], ROINAMESTR + ".png")
                if verbose:
                    print("%s: Saving %s\n" % (roicountStr, savename))
                imwrite(im=roi_out[imtype], uri=savename)
                this_roi_savenames[imtype] = savename

        if 'contours' in roi_out.keys():
            savename = os.path.join(
                save_directories['contours'], ROINAMESTR + ".csv")
            if verbose:
                print("%s: Saving %s\n" % (roicountStr, savename))
            contours_df = DataFrame(roi_out['contours'])
            contours_df.to_csv(savename)
            this_roi_savenames['contours'] = savename

        savenames.append(this_roi_savenames)

    return savenames

# %% =====================================================================
