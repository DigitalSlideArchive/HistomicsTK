"""
Created on Fri Jan 24 2020.

@author: mtageld
"""
import copy
import os
from itertools import combinations

import numpy as np
import pandas as pd
from imageio import imwrite

from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    _get_coords_from_element, _get_idxs_for_all_rois,
    _simple_add_element_to_roi, get_bboxes_from_slide_annotations,
    get_idxs_for_annots_overlapping_roi_by_bbox, get_image_from_htk_response,
    get_scale_factor_and_appendStr, parse_slide_annotations_into_tables,
    scale_slide_annotations)
from histomicstk.annotations_and_masks.annotations_to_masks_handler import (
    _get_roi_bounds_by_run_mode, _visualize_annotations_on_rgb)


def _sanity_checks(
        MPP, MAG, mode, bounds, idx_for_roi, get_rgb, get_visualization):

    # MPP precedes MAG
    if all([j is not None for j in (MPP, MAG)]):
        MAG = None

    # some sanity checks

    for mf in (MPP, MAG):
        if mf is not None:
            assert mf > 0, 'MPP or MAG must be positive.'

    if mode in ['wsi', 'min_bounding_box']:
        bounds = None
        idx_for_roi = None

    if idx_for_roi is not None:
        mode = 'polygonal_bounds'
    elif bounds is not None:
        mode = 'manual_bounds'

    assert mode in [
        'wsi', 'min_bounding_box', 'manual_bounds', 'polygonal_bounds'], \
        'mode %s not recognized' % mode

    if get_visualization:
        assert get_rgb, 'cannot get visualization without rgb.'

    return MPP, MAG, mode, bounds, idx_for_roi, get_rgb, get_visualization


def _keep_relevant_elements_for_roi(
        element_infos, sf, mode='manual_bounds',
        idx_for_roi=None, roiinfo=None):

    # This stores information about the ROI like bounds, slide_name, etc
    # Allows passing many parameters and good forward/backward compatibility
    if roiinfo is None:
        roiinfo = {}

    if mode != 'polygonal_bounds':
        # add to bounding boxes dataframe
        element_infos = pd.concat([element_infos, pd.DataFrame([{
            'xmin': int(roiinfo['XMIN'] * sf),
            'xmax': int(roiinfo['XMAX'] * sf),
            'ymin': int(roiinfo['YMIN'] * sf),
            'ymax': int(roiinfo['YMAX'] * sf),
        }])], ignore_index=True)
        idx_for_roi = element_infos.shape[0] - 1

    # isolate annotations that potentially overlap roi
    overlaps = get_idxs_for_annots_overlapping_roi_by_bbox(
        element_infos, idx_for_roi=idx_for_roi)
    if mode == 'polygonal_bounds':
        overlaps = overlaps + [idx_for_roi, ]
    elinfos_roi = element_infos.loc[overlaps, :]

    # update roiinfo -- remember, annotation elements can be
    # really large and extend beyond the bounds asked by the user.
    # since we're not parsing the polygons into mask form here, and
    # therefore we're not 'cropping' the polygons to the requested bounds,
    # we extend the requested bounds themselves to accommodate the overflowing
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

    # unique relevant annotation document indices & slice
    unique_annidxs = np.int32(np.unique(elinfos_roi.loc[:, 'annidx']))
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


def annotations_to_contours_no_mask(
        gc, slide_id, MPP=5.0, MAG=None, mode='min_bounding_box',
        bounds=None, idx_for_roi=None,
        slide_annotations=None, element_infos=None,
        linewidth=0.2, get_rgb=True, get_visualization=True, text=True):
    """Process annotations to get RGB and contours without intermediate masks.

    Parameters
    ----------
    gc : object
        girder client object to make requests, for example:
        gc = girder_client.GirderClient(apiUrl = APIURL)
        gc.authenticate(interactive=True)

    slide_id : str
        girder id for item (slide)

    MPP : float or None
        Microns-per-pixel -- best use this as it's more well-defined than
        magnification which is more scanner or manufacturer specific.
        MPP of 0.25 often roughly translates to 40x

    MAG : float or None
        If you prefer to use whatever magnification is reported in slide.
        If neither MPP or MAG is provided, everything is retrieved without
        scaling at base (scan) magnification.

    mode : str
        This specifies which part of the slide to get the mask from. Allowed
        modes include the following
        - wsi: get scaled up or down version of mask of whole slide
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
        scale_slide_annotations() to scale them up or down by sf BEFOREHAND.

    element_infos : pandas DataFrame.
        The columns annidx and elementidx
        encode the dict index of annotation document and element,
        respectively, in the original slide_annotations list of dictionaries.
        This can be obained by get_bboxes_from_slide_annotations() method.
        Make sure you have used scale_slide_annotations().

    linewidth : float
        visualization line width

    get_rgb: bool
        get rgb image?

    get_visualization : bool
        get overlaid annotation bounds over RGB for visualization

    text : bool
        add text labels to visualization?

    Returns
    --------
    dict
        Results dict containing one or more of the following keys
        - bounds: dict of bounds at scan magnification
        - rgb: (mxnx3 np array) corresponding rgb image
        - contours: dict
        - visualization: (mxnx3 np array) visualization overlay

    """
    MPP, MAG, mode, bounds, idx_for_roi, get_rgb, get_visualization = \
        _sanity_checks(
            MPP, MAG, mode, bounds, idx_for_roi,
            get_rgb, get_visualization)

    # calculate the scale factor
    sf, appendStr = get_scale_factor_and_appendStr(
        gc=gc, slide_id=slide_id, MPP=MPP, MAG=MAG)

    if slide_annotations is not None:
        assert element_infos is not None, 'must also provide element_infos'
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
    result = {}

    # get RGB
    if get_rgb:
        getStr = \
            '/item/%s/tiles/region?left=%d&right=%d&top=%d&bottom=%d&encoding=PNG' \
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
            rgb=rgb, contours_list=contours_list, linewidth=linewidth,
            text=text)

    return result


def combs_with_unique_products(low, high, k):
    prods = set()
    for comb in combinations(range(low, high), k):
        prod = np.prod(comb)
        if prod not in prods:
            yield comb
            prods.add(prod)


def contours_to_labeled_object_mask(
        contours, gtcodes, mode='object', verbose=False, monitorprefix=''):
    """Process contours to get and object segmentation labeled mask.

    Parameters
    ----------
    contours : DataFrame
        contours corresponding to annotation elements from the slide.
        All coordinates are relative to the mask that you want to output.
        The following columns are expected.
        - group: str, annotation group (ground truth label).
        - ymin: int, minimum y coordinate
        - ymax: int, maximum y coordinate
        - xmin: int, minimum x coordinate
        - xmax: int, maximum x coordinate
        - coords_x: str, vertex x coordinates comma-separated values
        - coords_y: str, vertex y coordinated comma-separated values

    gtcodes : DataFrame
        the ground truth codes and information dataframe.
        This is a dataframe that is indexed by the annotation group name
        and has the following columns.
        - group: str, group name of annotation, eg. mostly_tumor.
        - GT_code: int, desired ground truth code (in the mask).
        Pixels of this value belong to corresponding group (class).
        - color: str, rgb format. eg. rgb(255,0,0).

    mode : str
        run mode for getting masks. Must be in
        - object: get 3-channel mask where first channel encodes label
        (tumor, stroma, etc) while product of second and third
        channel encodes the object ID (i.e. individual contours)
        This is useful for object localization and segmentation tasks.
        - semantic: get a 1-channel mask corresponding to the first channel
        of the object mode.

    verbose : bool
        print to screen?

    monitorprefix : str
        prefix to add to printed statemens

    Returns
    -------
    np.array
        If mode is "object", this returns an (m, n, 3) np array of dtype uint8
        that can be saved as a png
        First channel: encodes label (can be used for semantic segmentation)
        Second & third channels: multiplication of second and third channel
        gives the object id (255 choose 2 = 32,385 max unique objects).
        This allows us to save into a convenient 3-channel png object labels
        and segmentation masks, which is more compact than traditional
        mask-rcnn save formats like having one channel per object and a
        separate csv file for object labels. This is also more convenient
        than simply saving things into pickled np array objects, and allows
        compatibility with data loaders that expect an image or mask.
        If mode is "semantic" only the labels (corresponding to first
        channel of the object mode) is output.
        ** IMPORTANT NOTE ** When you read this mask and decide to reconstruct
        the object codes, convert it to float32 so that the product doesn't
        saturate at 255.

    """
    def _process_gtcodes(gtcodesdf):
        # make sure ROIs are overlaid first
        # & assigned background class if relevant
        roi_groups = list(
            gtcodesdf.loc[gtcodesdf.loc[:, 'is_roi'] == 1, 'group'])
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
        raise Exception('Unknown run mode:', mode)

    # make sure roi is overlaid first + other processing
    gtcodes = _process_gtcodes(gtcodes)

    # unique combinations of number to be multiplied (second & third channel)
    # to be able to reconstruct the object ID when image is re-read
    object_code_comb = combs_with_unique_products(1, 256, 2)

    # Add annotations in overlay order
    overlay_orders = sorted(set(gtcodes.loc[:, 'overlay_order']))
    N_elements = contours.shape[0]

    # Make sure we don't run out of object encoding values.
    if N_elements > 17437:  # max unique products
        raise Exception('Too many objects!!')

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

        # get relevant infos and sort from largest to smallest (by bbox area)
        # so that the smaller elements are layered last. This helps partially
        # address issues describe in:
        # https://github.com/DigitalSlideArchive/HistomicsTK/issues/675
        elinfos_relevant = contours.loc[relIdxs, :].copy()
        elinfos_relevant.sort_values(
            'bbox_area', axis=0, ascending=False, inplace=True)

        # Go through elements and add to ROI mask
        for elId, elinfo in elinfos_relevant.iterrows():

            elNo += 1
            elcountStr = '%s: Overlay level %d: Element %d of %d: %s' % (
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
        return np.uint8(np.concatenate((
            labels_channel[..., None],
            objects_channel1[..., None],
            objects_channel2[..., None],
        ), -1))


def get_all_rois_from_slide_v2(
        gc, slide_id, GTCodes_dict, save_directories,
        annotations_to_contours_kwargs=None,
        mode='object', get_mask=True,
        slide_name=None, verbose=True, monitorprefix='',
        callback=None, callback_kwargs=None):
    """Get all ROIs for a slide without an intermediate mask form.

    This mainly relies on contours_to_labeled_object_mask(), which should
    be referred to for extra documentation.

    This can be run in either the "object" mode, whereby the saved masks
    are a three-channel png where first channel encodes class label (i.e.
    same as semantic segmentation) and the product of the values in the
    second and third channel encodes the object ID. Otherwise, the user
    may decide to run in the "semantic" mode and the resultant mask would
    consist of only one channel (semantic segmentation with no object
    differentiation).

    The difference between this and version 1, found at
    histomicstk.annotations_and_masks.annotations_to_masks_handler.
    get_all_rois_from_slide()
    is that this (version 2) gets the contours first, including cropping
    to wanted ROI boundaries and other processing using shapely, and THEN
    parses these into masks. This enables us to differentiate various objects
    to use the data for object localization or classification or segmentation
    tasks. If you would like to get semantic segmentation masks, i.e. you do
    not really care about individual objects, you can use either version 1
    or this method. They re-use much of the same code-base, but some edge
    cases maybe better handled by version 1. For example, since
    this version uses shapely first to crop, some objects may be incorrectly
    parsed by shapely. Version 1, using PIL.ImageDraw may not have these
    problems.

    Bottom line is: if you need semantic segmentation masks, it is probably
    safer to use version 1, whereas if you need object segmentation masks,
    this method should be used.

    Parameters
    ----------
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
        mask. Larger values means this annotation group is overlaid
        last and overwrites whatever overlaps it.
        - GT_code: int, desired ground truth code (in the mask)
        Pixels of this value belong to corresponding group (class)
        - is_roi: Flag for whether this group encodes an ROI
        - is_background_class: Flag, whether this group is the default
        fill value inside the ROI. For example, you may decide that
        any pixel inside the ROI is considered stroma.

    save_directories : dict
        paths to directories to save data. Each entry is a string, and the
        following keys are allowed
        - ROI: path to save masks (labeled images)
        - rgb: path to save rgb images
        - contours: path to save annotation contours
        - visualization: path to save rgb visualization overlays

    mode : str
        run mode for getting masks. Must me in
        - object: get 3-channel mask where first channel encodes label
        (tumor, stroma, etc) while product of second and third
        channel encodes the object ID (i.e. individual contours)
        This is useful for object localization and segmentation tasks.
        - semantic: get a 1-channel mask corresponding to the first channel
        of the object mode.

    get_mask : bool
        While the main purpose of this method IS to get object segmentation
        masks, it is conceivable that some users might just want to get
        the RGB and contours. Default is True.

    annotations_to_contours_kwargs : dict
        kwargs to pass to annotations_to_contours_no_mask()
        default values are assigned if specific parameters are not given.

    slide_name : str or None
        If not given, its inferred using a server request using girder client.

    verbose : bool
        Print progress to screen?

    monitorprefix : str
        text to prepend to printed statements

    callback : function
        a callback function to run on the roi dictionary output. This is
        internal, but if you really want to use this, make sure the callback
        can accept the following keys and that you do NOT assign them yourself
        gc, slide_id, slide_name, MPP, MAG, verbose, monitorprefix
        Also, this callback MUST *ONLY* return the roi dictionary, whether
        or not it is modified inside it. If it is modified inside the callback
        then the modified version is the one that will be saved to disk.

    callback_kwargs : dict
        kwargs to pass to callback, not including the mandatory kwargs
        that will be passed internally (mentioned earlier here).

    Returns
    --------
    list of dicts
        each entry contains the following keys
        mask - path to saved mask
        rgb - path to saved rgb image
        contours - path to saved annotation contours
        visualization - path to saved rgb visualization overlay

    """
    from pandas import DataFrame

    default_keyvalues = {
        'MPP': None, 'MAG': None,
        'linewidth': 0.2,
        'get_rgb': True, 'get_visualization': True,
    }

    # assign defaults if nothing given
    kvp = annotations_to_contours_kwargs or {}  # for easy referencing
    for k, v in default_keyvalues.items():
        if k not in kvp.keys():
            kvp[k] = v

    # convert to df and sanity check
    gtcodes_df = DataFrame.from_dict(GTCodes_dict, orient='index')
    if any(gtcodes_df.loc[:, 'GT_code'] <= 0):
        raise Exception('All GT_code must be > 0')

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
        GTCodes=gtcodes_df, element_infos=element_infos)

    savenames = []

    for roino, idx_for_roi in enumerate(idxs_for_all_rois):

        roicountStr = '%s: roi %d of %d' % (
            monitorprefix, roino + 1, len(idxs_for_all_rois))

        # get specified area
        roi_out = annotations_to_contours_no_mask(
            gc=gc, slide_id=slide_id,
            mode='polygonal_bounds', idx_for_roi=idx_for_roi,
            slide_annotations=slide_annotations,
            element_infos=element_infos, **kvp)

        # get corresponding mask (semantic or object)
        if get_mask:
            roi_out['mask'] = contours_to_labeled_object_mask(
                contours=DataFrame(roi_out['contours']),
                gtcodes=gtcodes_df,
                mode=mode, verbose=verbose, monitorprefix=roicountStr)

        # now run callback on roi_out
        if callback is not None:
            # these are 'compulsory' kwargs for the callback
            # since it will not have access to these otherwise
            callback_kwargs.update({
                'gc': gc,
                'slide_id': slide_id,
                'slide_name': slide_name,
                'MPP': kvp['MPP'],
                'MAG': kvp['MAG'],
                'verbose': verbose,
                'monitorprefix': roicountStr,
            })
            callback(roi_out, **callback_kwargs)

        # now save roi (rgb, vis, mask)

        this_roi_savenames = {}
        ROINAMESTR = '%s_left-%d_top-%d_bottom-%d_right-%d' % (
            slide_name,
            roi_out['bounds']['XMIN'], roi_out['bounds']['YMIN'],
            roi_out['bounds']['YMAX'], roi_out['bounds']['XMAX'])

        for imtype in ['mask', 'rgb', 'visualization']:
            if imtype in roi_out.keys():
                savename = os.path.join(
                    save_directories[imtype], ROINAMESTR + '.png')
                if verbose:
                    print('%s: Saving %s' % (roicountStr, savename))
                imwrite(im=roi_out[imtype], uri=savename)
                this_roi_savenames[imtype] = savename

        # save contours
        savename = os.path.join(
            save_directories['contours'], ROINAMESTR + '.csv')
        if verbose:
            print('%s: Saving %s\n' % (roicountStr, savename))
        contours_df = DataFrame(roi_out['contours'])
        contours_df.to_csv(savename)
        this_roi_savenames['contours'] = savename

        savenames.append(this_roi_savenames)

    return savenames
