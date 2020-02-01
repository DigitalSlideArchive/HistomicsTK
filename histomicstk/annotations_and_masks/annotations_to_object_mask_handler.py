# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 2020

@author: mtageld
"""
import copy
import numpy as np
from itertools import combinations
from histomicstk.annotations_and_masks.annotation_and_mask_utils import \
    get_idxs_for_annots_overlapping_roi_by_bbox, \
    get_scale_factor_and_appendStr, scale_slide_annotations, \
    get_bboxes_from_slide_annotations, get_image_from_htk_response, \
    parse_slide_annotations_into_tables, _get_coords_from_element, \
    _simple_add_element_to_roi
from histomicstk.annotations_and_masks.annotations_to_masks_handler import \
    _get_roi_bounds_by_run_mode, _visualize_annotations_on_rgb

# %%===========================================================================


def _sanity_checks(
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


def _keep_relevant_elements_for_roi(
        element_infos, sf, mode='manual_bounds',
        idx_for_roi=None, roiinfo=None):

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
        element_infos, idx_for_roi=idx_for_roi)
    if mode == "polygonal_bounds":
        overlaps = overlaps + [idx_for_roi, ]
    elinfos_roi = element_infos.loc[overlaps, :]

    # update roiinfo -- remember, annotation elements can be
    # really large and extend beyond the bounds asked by the user.
    # since we're not parsing the polygons into mask form here, and
    # therefore we're not 'cropping' the polygons to the requested bounds,
    # we extend the requested bounds themselves to accomodate the overflowing
    # annotations.
    roiinfo['XMIN'] = int(np.min(elinfos_roi.xmin))
    roiinfo['YMIN'] = int(np.min(elinfos_roi.ymin))
    roiinfo['XMAX'] = int(np.max(elinfos_roi.xmax))
    roiinfo['YMAX'] = int(np.max(elinfos_roi.ymax))
    roiinfo['BBOX_WIDTH'] = roiinfo['XMAX'] - roiinfo['XMIN']
    roiinfo['BBOX_HEIGHT'] = roiinfo['YMAX'] - roiinfo['YMIN']

    # scale back coords
    roiinfo = {k: int(v / sf) for k, v in roiinfo.items()}

    return elinfos_roi, roiinfo


def _trim_slide_annotations_to_roi(annotations, elinfos_roi):

    # unique relevent annotation document indices & slice
    unique_annidxs = np.int32(np.unique(elinfos_roi.loc[:, "annidx"]))
    annotations_slice = np.array(annotations)[unique_annidxs].tolist()

    # anno is index relative to unique_annidxs, while
    # annidx is index relative to original slide annotations
    for anno, annidx in enumerate(unique_annidxs):

        # indices of relevant elements in this annotation doc
        eleidxs = np.int32(elinfos_roi.loc[
            elinfos_roi.loc[:, 'annidx'] == annidx, 'elementidx'])

        # slice relevant elements
        elements_original = annotations_slice[anno]['annotation']['elements']
        annotations_slice[anno]['annotation']['elements'] = np.array(
            elements_original)[eleidxs].tolist()

    return annotations_slice

# %%===========================================================================


def annotations_to_contours_no_mask(
        gc, slide_id, MPP=5.0, MAG=None, mode='min_bounding_box',
        bounds=None, idx_for_roi=None,
        slide_annotations=None, element_infos=None,
        linewidth=0.2, get_rgb=True, get_visualization=True):
    """Process annotations to get RGB and contours without intermediate masks.

    Parameters
    ----------
    gc
    slide_id
    MPP
    MAG
    mode
    bounds
    idx_for_roi
    slide_annotations
    element_infos
    linewidth
    get_rgb
    get_visualization

    Returns
    -------

    """
    MPP, MAG, mode, bounds, idx_for_roi, get_rgb, get_visualization = \
        _sanity_checks(
            MPP, MAG, mode, bounds, idx_for_roi,
            get_rgb, get_visualization)

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

    # Determine get region based on run mode, keeping in mind that it
    # must be at BASE MAGNIFICATION coordinates before it is passed
    # on to get_mask_from_slide()
    # if mode != 'polygonal_bound':
    bounds = _get_roi_bounds_by_run_mode(
        gc=gc, slide_id=slide_id, mode=mode, bounds=bounds,
        element_infos=element_infos, idx_for_roi=idx_for_roi, sf=sf)

    # only keep relevant elements and get uncropped bounds
    elinfos_roi, uncropped_bounds = _keep_relevant_elements_for_roi(
        element_infos, sf=sf, mode=mode, idx_for_roi=idx_for_roi,
        roiinfo=copy.deepcopy(bounds))

    # find relevant portion from slide annotations to use
    # (with overflowing beyond edge)
    annotations_slice = _trim_slide_annotations_to_roi(
        copy.deepcopy(slide_annotations), elinfos_roi=elinfos_roi)

    # get roi polygon vertices
    rescaled_bounds = {k: int(v * sf) for k, v in bounds.items()}
    if mode == 'polygonal_bounds':
        roi_coords = _get_coords_from_element(copy.deepcopy(
            slide_annotations[int(element_infos.loc[idx_for_roi, 'annidx'])]
            ['annotation']['elements']
            [int(element_infos.loc[idx_for_roi, 'elementidx'])]))
        cropping_bounds = None
    else:
        roi_coords = None
        cropping_bounds = rescaled_bounds

    # tabularize to use contours
    _, contours_df = parse_slide_annotations_into_tables(
        annotations_slice, cropping_bounds=cropping_bounds,
        cropping_polygon_vertices=roi_coords,
        use_shapely=mode in ('manual_bounds', 'polygonal_bounds'),
    )
    contours_list = contours_df.to_dict(orient='records')

    # Final bounds (relative to slide at base magnification)
    bounds = {k: int(v / sf) for k, v in rescaled_bounds.items()}
    result = dict()

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

    # Assign to results
    result.update({
        'contours': contours_list,
        'bounds': bounds,
    })

    # get visualization of annotations on RGB
    if get_visualization:
        result['visualization'] = _visualize_annotations_on_rgb(
            rgb=rgb, contours_list=contours_list, linewidth=linewidth)

    return result

# %%===========================================================================


def contours_to_labeled_object_mask(
        contours, gtcodes, mode='object', verbose=False, monitorprefix=''):
    """Process contours to get and object segmentation labeled mask.

    Parameters
    ----------
    contours : DataFrame
    gtcodes : DataFrame
    mode : str
    verbose : bool
    monitorprefix : str

    Returns
    -------
    np.array
        An (m, n, 3) np array that can be saved as a png
        - First channel: encodes label (can be used for semantic segmentation)
        - Second & third channels: multiplication of second and third channel
          gives the object id (255 choose 2 = 32,385 max unique objects)

    """
    def _process_gtcodes(gtcodesdf):
        # make sure ROIs are overlayed first
        # & assigned background class if relevant
        roi_groups = list(
            gtcodesdf.loc[gtcodesdf.loc[:, 'is_roi'] == 1, "group"])
        roi_order = np.min(gtcodesdf.loc[:, 'overlay_order']) - 1
        bck_classes = gtcodesdf.loc[
            gtcodesdf.loc[:, 'is_background_class'] == 1, :]
        for roi_group in roi_groups:
            gtcodesdf.loc[roi_group, 'overlay_order'] = roi_order
            if bck_classes.shape[0] > 0:
                gtcodesdf.loc[
                    roi_group, 'GT_code'] = bck_classes.iloc[0, :]['GT_code']
        return gtcodesdf

    if mode not in ['semantic', 'object']:
        raise Exception("Unknown run mode:", mode)

    # make sure roi is overlayed first + other processing
    gtcodes = _process_gtcodes(gtcodes)

    # unique combinations of number to be multiplied (second & third channel)
    # to be able to reconstruct the object ID when image is re-read
    object_code_comb = combinations(range(1, 256), 2)

    # Add annotations in overlay order
    overlay_orders = sorted(set(gtcodes.loc[:, 'overlay_order']))
    N_elements = contours.shape[0]

    # Make sure we don't run out of object encoding values.
    if N_elements > 32358:
        raise Exception("Too many objects!!")

    # Add roiinfo & init roi
    roiinfo = {
        'XMIN': int(np.min(contours.xmin)),
        'YMIN': int(np.min(contours.ymin)),
        'XMAX': int(np.max(contours.xmax)),
        'YMAX': int(np.max(contours.ymax)),
    }
    roiinfo['BBOX_WIDTH'] = roiinfo['XMAX'] - roiinfo['XMIN']
    roiinfo['BBOX_HEIGHT'] = roiinfo['YMAX'] - roiinfo['YMIN']

    # init channels
    labels_channel = np.zeros(
            (roiinfo['BBOX_HEIGHT'], roiinfo['BBOX_WIDTH']), dtype=np.uint8)
    if mode == 'object':
        objects_channel1 = labels_channel.copy()
        objects_channel2 = labels_channel.copy()

    elNo = 0
    for overlay_level in overlay_orders:

        # get indices of relevant groups
        relevant_groups = list(gtcodes.loc[
            gtcodes.loc[:, 'overlay_order'] == overlay_level, 'group'])
        relIdxs = []
        for group_name in relevant_groups:
            relIdxs.extend(list(contours.loc[
                contours.group == group_name, :].index))

        # get relevnt infos and sort from largest to smallest (by bbox area)
        # so that the smaller elements are layered last. This helps partially
        # address issues describe in:
        # https://github.com/DigitalSlideArchive/HistomicsTK/issues/675
        elinfos_relevant = contours.loc[relIdxs, :].copy()
        elinfos_relevant.sort_values(
            'bbox_area', axis=0, ascending=False, inplace=True)

        # Go through elements and add to ROI mask
        for elId, elinfo in elinfos_relevant.iterrows():

            elNo += 1
            elcountStr = "%s: Overlay level %d: Element %d of %d: %s" % (
                monitorprefix, overlay_level, elNo, N_elements,
                elinfo['group'])
            if verbose:
                print(elcountStr)

            # Add element to labels channel
            labels_channel, element = _simple_add_element_to_roi(
                elinfo=elinfo, ROI=labels_channel, roiinfo=roiinfo,
                GT_code=gtcodes.loc[elinfo['group'], 'GT_code'],
                verbose=verbose, monitorPrefix=elcountStr)

            if (element is not None) and (mode == 'object'):

                object_code = next(object_code_comb)

                # Add element to object (instance) channel 1
                objects_channel1, _ = _simple_add_element_to_roi(
                    elinfo=elinfo, ROI=objects_channel1, roiinfo=roiinfo,
                    GT_code=object_code[0], element=element,
                    verbose=verbose, monitorPrefix=elcountStr)

                # Add element to object (instance) channel 2
                objects_channel2, _ = _simple_add_element_to_roi(
                    elinfo=elinfo, ROI=objects_channel2, roiinfo=roiinfo,
                    GT_code=object_code[1], element=element,
                    verbose=verbose, monitorPrefix=elcountStr)

    # Now concat to get final product
    # If the mode is object segmentation, we get an np array where
    # - First channel: encodes label (can be used for semantic segmentation)
    # - Second & third channels: multiplication of second and third channel
    #       gives the object id (255 choose 2 = 32,385 max unique objects)
    # This enables us to later save these masks in convenient compact
    # .png format

    if mode == 'semantic':
        return labels_channel
    else:
        return np.concatenate((
            labels_channel[..., None],
            objects_channel1[..., None],
            objects_channel2[..., None],
            ), -1)

# %%===========================================================================
