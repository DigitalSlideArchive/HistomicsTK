"""
Created on Sun Aug 11 22:30:06 2019.

@author: tageldim

"""
import copy
import warnings
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry.polygon import Polygon

Image.MAX_IMAGE_PIXELS = None


def get_image_from_htk_response(resp):
    """Given a histomicsTK girder response, get np array image.

    Parameters
    ----------
    resp : object
           response from server request

    Returns
    -------
    Pillow Image object
        a pillow Image object of the image

    """
    image_content = BytesIO(resp.content)
    image_content.seek(0)
    image = Image.open(image_content)
    image = image.convert('RGB')
    return np.uint8(image)


def scale_slide_annotations(slide_annotations, sf):
    """Scales up or down annotations in a slide.

    Works in place, but returns slide_annotations anyways.

    Parameters
    ----------
    slide_annotations : list of dicts
        response from server request

    sf : float
        scale factor to multiply coordinates

    Returns
    -------
    list of dicts

    """
    if sf == 1.0:
        return slide_annotations

    for _annidx, ann in enumerate(slide_annotations):
        for element in ann['annotation']['elements']:
            for key in element.keys():

                if key in ['height', 'width']:
                    element[key] = int(element[key] * sf)

                elif key in ['center', 'points']:
                    element[key] = (np.array(element[key]) * sf).astype(int).tolist()
    return slide_annotations


def get_scale_factor_and_appendStr(gc, slide_id, MPP=None, MAG=None):
    """Get how much is request region smaller than base.

    This also gets the string to append to server request for getting rgb.

    Parameters
    ----------
    gc : Girder client instance
        gc should be authoenticated.
    slide_id : str
        girder id of slide
    MPP : float or None
        Microns-per-pixel -- best use this as it's more well-defined than
        magnification which is more scanner/manufacturer specific.
        MPP of 0.25 often roughly translates to 40x
    MAG : float or None
        If you prefer to use whatever magnification is reported in slide.
        If neither MPP or MAG is provided, everything is retrieved without
        scaling at base (scan) magnification.

    Returns
    -------
    float
        how much smaller (0.1 means 10x smaller) is requested image
        compared to scan magnification (slide coordinates)
    str
        string to appnd to server request for getting slide region

    """
    slide_info = gc.get('/item/%s/tiles' % slide_id)

    if (MPP is not None) and (slide_info['mm_x'] is not None):
        mm = 0.001 * MPP
        sf = slide_info['mm_x'] / mm
        appendStr = '&mm_x=%.8f&mm_y=%.8f' % (mm, mm)

    elif (MAG is not None) and (slide_info['magnification'] is not None):
        sf = MAG / slide_info['magnification']
        appendStr = '&magnification=%.8f' % MAG

    else:
        warnings.warn(
            'NO SLIDE MAGNIFICATION FOUND; BASE MAGNIFICATION USED!',
            RuntimeWarning, stacklevel=1)
        sf = 1.0
        appendStr = ''

    return sf, appendStr


def rotate_point_list(point_list, rotation, center=(0, 0)):
    """Rotate a certain point list around a central point. Modified from.

    javascript version at: https://github.com/girder/large_image/blob/master/
    ... web_client/annotations/rotate.js.

    Parameters
    ----------
    point_list : list of tuples
        (x,y) coordinates

    rotation : float
        degrees (in radians)

    center : tuple of ints
        central point coordinates

    Returns
    -------
    list of tuples

    """
    point_list_rotated = []

    for point in point_list:

        cos = np.cos(rotation)
        sin = np.sin(rotation)
        x = point[0] - center[0]
        y = point[1] - center[1]

        point_list_rotated.append((
            int(x * cos - y * sin + center[0]),
            int(x * sin + y * cos + center[1])))

    return point_list_rotated


def get_rotated_rectangular_coords(
        roi_center, roi_width, roi_height, roi_rotation=0):
    """Given data on rectangular ROI center/width/height/rotation.

    Get the unrotated abounding box coordinates around rotated ROI. This
    of course is applicable to any rotated rectangular annotation.

    Parameters
    ----------
    roi_center : tuple or list
        (x, y) format
    roi_width : float
    roi_height : float
    roi_rotation : float

    Returns
    -------
    dict
        includes roi corners (x, y) and bounds

    """
    # Get false bounds
    x_min = roi_center[0] - int(roi_width / 2)
    x_max = roi_center[0] + int(roi_width / 2)
    y_min = roi_center[1] - int(roi_height / 2)
    y_max = roi_center[1] + int(roi_height / 2)

    # Get pseudo-corners of rectangle (before rotation)
    roi_corners_false = [
        (x_min, y_min), (x_max, y_min),
        (x_max, y_max), (x_min, y_max)]

    # Get *true* coordinates of rectangle corners
    roi_corners = rotate_point_list(roi_corners_false,
                                    rotation=roi_rotation,
                                    center=roi_center)
    roi_corners = np.array(roi_corners, dtype=np.int32)

    # pack into dict
    roi_info = {
        'roi_corners': roi_corners,
        'x_min': roi_corners[:, 0].min(),
        'x_max': roi_corners[:, 0].max(),
        'y_min': roi_corners[:, 1].min(),
        'y_max': roi_corners[:, 1].max(),
    }

    return roi_info


def get_bboxes_from_slide_annotations(slide_annotations):
    """Given a slide annotation list, gets information on bounding boxes.

    Parameters
    ----------
    slide_annotations : list of dicts
        response from server request

    Returns
    -------
    Pandas DataFrame
        The columns annidx and elementidx encode the
        dict index of annotation document and element, respectively, in the
        original slide_annotations list of dictionaries

    """
    from pandas import DataFrame

    element_infos = DataFrame(columns=[
        'annidx', 'elementidx', 'type', 'group',
        'xmin', 'xmax', 'ymin', 'ymax'])

    for annidx, ann in enumerate(slide_annotations):
        for elementidx, element in enumerate(ann['annotation']['elements']):

            elno = element_infos.shape[0]
            element_infos.loc[elno, 'annidx'] = annidx
            element_infos.loc[elno, 'elementidx'] = elementidx
            element_infos.loc[elno, 'type'] = element['type']

            # get bounds
            if element['type'] == 'polyline':
                coords = np.array(element['points'])[:, :-1]
                xmin, ymin = (int(j) for j in np.min(coords, axis=0))
                xmax, ymax = (int(j) for j in np.max(coords, axis=0))

            elif element['type'] == 'rectangle':
                roiinfo = get_rotated_rectangular_coords(
                    roi_center=element['center'],
                    roi_width=element['width'],
                    roi_height=element['height'],
                    roi_rotation=element['rotation'])
                xmin, ymin = roiinfo['x_min'], roiinfo['y_min']
                xmax, ymax = roiinfo['x_max'], roiinfo['y_max']

            else:
                continue

            # add group or infer from label
            if 'group' in element.keys():
                element_infos.loc[elno, 'group'] = element['group']
            elif 'label' in element.keys():
                element_infos.loc[elno, 'group'] = element['label']['value']

            element_infos.loc[elno, 'xmin'] = xmin
            element_infos.loc[elno, 'xmax'] = xmax
            element_infos.loc[elno, 'ymin'] = ymin
            element_infos.loc[elno, 'ymax'] = ymax
            element_infos.loc[elno, 'bbox_area'] = int(
                (ymax - ymin) * (xmax - xmin))

    return element_infos


def _get_coords_from_element(element):

    # get bounds
    if element['type'] == 'polyline':
        coords = np.int32(element['points'])[:, :-1]

    elif element['type'] == 'rectangle':
        roiinfo = get_rotated_rectangular_coords(
            roi_center=element['center'],
            roi_width=element['width'],
            roi_height=element['height'],
            roi_rotation=element['rotation'])
        coords = roiinfo['roi_corners']  # for rotated rectangles
        if element['rotation'] != 0:
            element['type'] = 'polyline'

    elif element['type'] == 'point':
        xmin = xmax = int(element['center'][0])
        ymin = ymax = int(element['center'][1])
        coords = np.array(
            [(xmin, ymin), (xmax, ymin), (xmax, ymax),
             (xmin, ymax), (xmin, ymin)], dtype='int32')

    else:
        msg = 'Unsupported element type:'
        raise Exception(msg, element['type'])

    return coords


def _maybe_crop_polygon(vertices, bounds_polygon):
    """Crop bounds to desired area using shapely polygons."""
    all_vertices = []

    # First, we get the polygon or polygons which result from
    # intersection. Keep in mind that a particular annotation may
    # "ecit then enter" the cropping ROI, so it may be split into
    # two or more polygons by the splitting process. That's which
    # this method's input is one set of vertices, but after cropping
    # it may return one or more sets of vertices
    try:
        elpoly = Polygon(vertices).buffer(0)
        polygon = bounds_polygon.intersection(elpoly)

        if polygon.geom_type in ('Polygon', 'LineString'):
            polygons_to_add = [polygon]
        else:
            polygons_to_add = [
                p for p in polygon if p.geom_type == 'Polygon']

    except Exception as e:
        # if weird shapely errors -->
        # ignore this polygon collection altogether
        print(e.__repr__())
        return all_vertices

    # Second, once we have the shapely polygons, assuming no errors,
    # we parse into usable coordinates that we can add to the table
    for poly in polygons_to_add:
        try:
            if polygon.area > 2:
                all_vertices.append(np.array(
                    poly.exterior.xy, dtype=np.int32).T)
        except Exception as e:
            # weird shapely errors --> ignore this polygon
            print(e.__repr__())

    return all_vertices


def _parse_coords_to_str(vertices):
    return (
        ','.join(str(j) for j in vertices[:, 0]),
        ','.join(str(j) for j in vertices[:, 1]))


def _add_element_to_final_df(vertices, cfg):
    """Add a single element to the final dataframe.

    Note that we wrap this into a method so that when (if) we split an
    annotation element into multiple polygons, we can add these
    separately.
    """
    elno = cfg.element_infos.shape[0]

    # Add element information to element dataframe
    cfg.element_infos.loc[elno, 'annidx'] = cfg.annidx
    cfg.element_infos.loc[elno, 'annotation_girder_id'] = cfg.ann['_id']
    cfg.element_infos.loc[elno, 'elementidx'] = cfg.elementidx
    cfg.element_infos.loc[elno, 'element_girder_id'] = cfg.element['id']
    cfg.element_infos.loc[elno, 'color'] = str(cfg.element['lineColor'])

    # Now we can add offset to ensure coordinates are relative to the
    # cropping bounds (i.e. they would correspond to an RGB image
    # of the same region and could be used to create a mask or
    # to encode object boundaries etc
    if (cfg.cropping_bounds is not None) or (
            cfg.cropping_polygon_vertices is not None):
        vertices[:, 0] = vertices[:, 0] - cfg.x_shift
        vertices[:, 1] = vertices[:, 1] - cfg.y_shift

    # get bounds for this polygon. Remember, these may have been
    # changed after it was cropped using shapely
    xmin, ymin = np.min(vertices, axis=0)
    xmax, ymax = np.max(vertices, axis=0)

    # parse to string for inclusion in pd dataframe
    x_coords, y_coords = _parse_coords_to_str(vertices)

    # add group or infer from label
    if 'group' in cfg.element.keys():
        cfg.element_infos.loc[elno, 'group'] = str(cfg.element['group'])
    elif 'label' in cfg.element.keys():
        cfg.element_infos.loc[elno, 'group'] = str(
            cfg.element['label']['value'])

    # add label or infer from group
    if 'label' in cfg.element.keys():
        cfg.element_infos.loc[elno, 'label'] = str(
            cfg.element['label']['value'])
    elif 'group' in cfg.element.keys():
        cfg.element_infos.loc[elno, 'label'] = cfg.element_infos.loc[
            elno, 'group']

    cfg.element_infos.loc[elno, 'type'] = str(cfg.element['type'])
    cfg.element_infos.loc[elno, 'xmin'] = int(xmin)
    cfg.element_infos.loc[elno, 'xmax'] = int(xmax)
    cfg.element_infos.loc[elno, 'ymin'] = int(ymin)
    cfg.element_infos.loc[elno, 'ymax'] = int(ymax)
    cfg.element_infos.loc[elno, 'bbox_area'] = int(
        (ymax - ymin) * (xmax - xmin))
    cfg.element_infos.loc[elno, 'coords_x'] = x_coords
    cfg.element_infos.loc[elno, 'coords_y'] = y_coords


def parse_slide_annotations_into_tables(
        slide_annotations, cropping_bounds=None,
        cropping_polygon_vertices=None, use_shapely=False):
    """Given a slide annotation list, parse into convenient tabular format.

    If the annotation is a point, then it is just treated as if it is a
    rectangle with zero area (i.e. xmin=xmax). Rotated rectangles are treated
    as polygons for simplicity.

    Parameters
    ----------
    slide_annotations : list of dicts
        response from server request

    cropping_bounds : dict or None
        if given, must have keys XMIN, XMAX, YMIN, YMAX. These are the
        bounds to which the polygons may be cropped using shapely,
        if the param use_shapely is True. Otherwise, the polygon
        coordinates are just shifted relative to these bounds without
        actually cropping.

    cropping_polygon_vertices : nd array or None
        if given, is an (m, 2) nd array of vertices to crop bounds.
        if the param use_shapely is True. Otherwise, the polygon
        coordinates are just shifted relative to these bounds without
        actually cropping.

    use_shapely : bool
        see cropping_bounds description.

    Returns
    -------
    Pandas DataFrame
        Summary of key properties of the annotation documents. It has the
        following columns:
        - annotation_girder_id
        - _modelType
        - _version
        - itemId
        - created
        - creatorId
        - public
        - updated
        - updatedId
        - groups
        - element_count
        - element_details

    Pandas DataFrame

        The individual annotation elements (polygons, points, rectangles).
        The columns annidx and elementidx encode the dict index of annotation
        document and element, respectively, in the original slide_annotations
        list of dictionaries. It has the following columns:

        - annidx
        - annotation_girder_id
        - elementidx
        - element_girder_id
        - type
        - group
        - label
        - color
        - xmin
        - xmax
        - ymin
        - ymax
        - bbox_area
        - coords_x
        - coords_y

    """
    from pandas import DataFrame

    # we use this object to pass params to split method into sub-methods
    # and avoid annoying linting ("method too complex") issue
    class Cfg:
        def __init__(self):
            pass
    cfg = Cfg()
    cfg.cropping_bounds = cropping_bounds
    cfg.cropping_polygon_vertices = cropping_polygon_vertices
    cfg.use_shapely = use_shapely

    cfg.annotation_infos = DataFrame(columns=[
        'annotation_girder_id', '_modelType', '_version',
        'itemId', 'created', 'creatorId',
        'public', 'updated', 'updatedId',
        'groups', 'element_count', 'element_details',
    ])

    cfg.element_infos = DataFrame(columns=[
        'annidx', 'annotation_girder_id',
        'elementidx', 'element_girder_id',
        'type', 'group', 'label', 'color',
        'xmin', 'xmax', 'ymin', 'ymax', 'bbox_area',
        'coords_x', 'coords_y',
    ])

    if cfg.cropping_bounds is not None:
        assert cfg.cropping_polygon_vertices is None, \
            'either give cropping bounds or vertices, not both'
        xmin, xmax, ymin, ymax = (
            cfg.cropping_bounds['XMIN'], cfg.cropping_bounds['XMAX'],
            cfg.cropping_bounds['YMIN'], cfg.cropping_bounds['YMAX'])
        bounds_polygon = Polygon(np.array([
            (xmin, ymin), (xmax, ymin),
            (xmax, ymax), (xmin, ymax), (xmin, ymin),
        ], dtype='int32'))
        cfg.x_shift = xmin
        cfg.y_shift = ymin

    elif cfg.cropping_polygon_vertices is not None:
        bounds_polygon = Polygon(np.int32(cfg.cropping_polygon_vertices))
        cfg.x_shift, cfg.y_shift = np.min(cfg.cropping_polygon_vertices, 0)

    # go through annotation elements and add as needed
    for cfg.annidx, cfg.ann in enumerate(slide_annotations):

        annno = cfg.annotation_infos.shape[0]

        # Add annotation document info to annotations dataframe

        cfg.annotation_infos.loc[annno, 'annotation_girder_id'] = cfg.ann['_id']

        for key in [
                '_modelType', '_version',
                'itemId', 'created', 'creatorId',
                'public', 'updated', 'updatedId']:
            if key in cfg.ann:
                cfg.annotation_infos.loc[annno, key] = cfg.ann[key]

        if 'groups' in cfg.ann:
            cfg.annotation_infos.loc[annno, 'groups'] = str(cfg.ann['groups'])

        if '_elementQuery' in cfg.ann:
            cfg.annotation_infos.loc[annno, 'element_count'] = cfg.ann[
                '_elementQuery']['count']
            cfg.annotation_infos.loc[annno, 'element_details'] = cfg.ann[
                '_elementQuery']['details']

        for idx, element in enumerate(cfg.ann['annotation']['elements']):
            cfg.elementidx = idx
            cfg.element = element
            coords = _get_coords_from_element(copy.deepcopy(element))

            # crop using shapely to desired bounds if needed
            # IMPORTANT: everything till this point needs to be
            # relative to the whole slide image
            if ((cfg.cropping_bounds is None) and
                    (cfg.cropping_polygon_vertices is None)) \
                    or (element['type'] == 'point') \
                    or (not use_shapely):
                all_coords = [coords]
            else:
                all_coords = _maybe_crop_polygon(coords, bounds_polygon)

            # now add polygons one by one
            for vertices in all_coords:
                _add_element_to_final_df(vertices, cfg=cfg)

    return cfg.annotation_infos, cfg.element_infos


def np_vec_no_jit_iou(bboxes1, bboxes2):
    """Fast, vectorized IoU.

    Source: https://medium.com/@venuktan/vectorized-intersection-over-union ...
            -iou-in-numpy-and-tensor-flow-4fa16231b63d

    Parameters
    ----------
    bboxes1 : np array
        columns encode bounding box corners xmin, ymin, xmax, ymax
    bboxes2 : np array
        same as bboxes 1

    Returns
    -------
    np array
        IoU values for each pair from bboxes1 & bboxes2

    """
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou


def _get_idxs_for_all_rois(GTCodes, element_infos):
    """Get indices of ROIs within the element_infos dataframe.

    (Internal)

    """
    roi_labels = list(GTCodes.loc[GTCodes.loc[:, 'is_roi'] == 1, 'group'])
    idxs_for_all_rois = []
    for idx, elinfo in element_infos.iterrows():
        if elinfo.group in roi_labels:
            idxs_for_all_rois.append(idx)
    return idxs_for_all_rois


def get_idxs_for_annots_overlapping_roi_by_bbox(
        element_infos, idx_for_roi, iou_thresh=0.0):
    """Find indices of **potentially** included annotations within the ROI.

    We say "potentially" because this uses the IoU of roi and annotation as a
    fast indicator of potential inclusion. This helps dramatically scale down
    the number of annotations to look through. Later on, a detailed look at
    whether the annotation polygons actually overlap the ROI can be done.

    Parameters
    ----------
    element_infos : pandas DataFrame
        result from running get_bboxes_from_slide_annotations()
    idx_for_roi : int
        index for roi annotation within the element_infos DF
    iou_thresh : float
        overlap threshold to be considered within ROI

    Returns
    -------
    list
        indices relative to element_infos

    """
    bboxes = np.array(
        element_infos.loc[:, ['xmin', 'ymin', 'xmax', 'ymax']],
        dtype='float')
    iou = np_vec_no_jit_iou(bboxes[idx_for_roi, :][None, ...], bboxes2=bboxes)
    iou = np.concatenate((np.arange(iou.shape[1])[None, ...], iou))
    iou = iou[:, iou[1, :] > iou_thresh].astype(int)

    overlaps = set(iou[0, :].tolist()) - {idx_for_roi}

    return list(overlaps)


def create_mask_from_coords(coords):
    """Create a binary mask from given vertices coordinates.

    Source: This is modified from code by Juan Carlos from David Gutman Lab.

    Parameters
    ----------
    coords : np array
        must be in the form (e.g. ([x1,y1],[x2,y2],[x3,y3],.....,[xn,yn])),
        where xn and yn corresponds to the nth vertex coordinate.

    Returns
    -------
    np array
        binary mask

    """
    polygon = coords.copy()

    # use the smallest bounding region, calculated from vertices
    min_x, min_y = np.min(polygon, axis=0)
    max_x, max_y = np.max(polygon, axis=0)
    # get the new width and height
    width = int(max_x - min_x)
    height = int(max_y - min_y)
    # shift all vertices to account for location of the smallest bounding box
    polygon[:, 0] = polygon[:, 0] - min_x
    polygon[:, 1] = polygon[:, 1] - min_y

    # convert to tuple form for masking function (nd array does not work)
    vertices_tuple = tuple(map(tuple, polygon))
    # make the mask
    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon(vertices_tuple, outline=1, fill=1)

    return np.array(img, dtype='int32')


def _get_element_mask(elinfo, slide_annotations):
    """Get coordinates and mask for annotation element.

    (Internal)

    """
    element = slide_annotations[int(elinfo['annidx'])][
        'annotation']['elements'][int(elinfo['elementidx'])]

    if elinfo['type'] == 'polyline':
        coords = np.array(element['points'])[:, :-1]
    elif elinfo['type'] == 'rectangle':
        infoDict = get_rotated_rectangular_coords(
            roi_center=element['center'], roi_width=element['width'],
            roi_height=element['height'], roi_rotation=element['rotation'])
        coords = infoDict['roi_corners']
    else:
        msg = 'cannot create mask from point annotation!'
        raise Exception(msg)

    mask = create_mask_from_coords(coords)
    return coords, mask


def _add_element_to_roi(elinfo, ROI, GT_code, element_mask, roiinfo):
    """Add single polygon to ROI given mask (Internal)."""
    ymin = int(elinfo.ymin - roiinfo['YMIN'])
    ymax = int(elinfo.ymax - roiinfo['YMIN'])
    xmin = int(elinfo.xmin - roiinfo['XMIN'])
    xmax = int(elinfo.xmax - roiinfo['XMIN'])
    patch = ROI[ymin:ymax, xmin:xmax]

    # to account for non-integer pixels
    if element_mask.shape != patch.shape:
        element_mask = np.pad(element_mask, pad_width=(
            (patch.shape[0] - element_mask.shape[0], 0),
            (patch.shape[1] - element_mask.shape[1], 0)), mode='constant')

    # add to ROI mask
    patch[element_mask > 0] = GT_code
    ROI[ymin:ymax, xmin:xmax] = patch.copy()

    element = {
        'mask': element_mask,
        'xmin': xmin, 'xmax': xmax,
        'ymin': ymin, 'ymax': ymax,
    }

    return ROI, element


def _get_and_add_element_to_roi(
        elinfo, slide_annotations, ROI, roiinfo, roi_polygon, GT_code,
        use_shapely=True, verbose=True, monitorPrefix=''):
    """Get element coords and mask and add to ROI (Internal)."""
    try:
        coords, element_mask = _get_element_mask(
            elinfo=elinfo, slide_annotations=slide_annotations)

        ADD_TO_ROI = True

        # ignore if outside ROI (precise)
        if use_shapely:
            el_polygon = Polygon(coords)
            if el_polygon.distance(roi_polygon) > 2:
                if verbose:
                    print('%s: OUTSIDE ROI.' % monitorPrefix)
                ADD_TO_ROI = False

        # Add element to ROI mask
        if ADD_TO_ROI:
            ROI, _ = _add_element_to_roi(
                elinfo=elinfo, ROI=ROI, GT_code=GT_code,
                element_mask=element_mask, roiinfo=roiinfo)

    except Exception as e:
        if verbose:
            print('%s: ERROR! (see below)' % monitorPrefix)
            print(e)
    return ROI


def delete_annotations_in_slide(gc, slide_id):
    """Delete all annotations in a slide."""
    existing_annotations = gc.get('/annotation/item/' + slide_id)
    for ann in existing_annotations:
        gc.delete('/annotation/%s' % ann['_id'])


def _simple_add_element_to_roi(
        elinfo, ROI, roiinfo, GT_code, element=None,
        verbose=True, monitorPrefix=''):
    """Get element coords and mask and add to ROI (Internal)."""

    def _process_coords(k):
        return np.array([int(j) for j in elinfo[k].split(',')])[..., None]

    try:
        if element is None:
            coords = np.concatenate([
                _process_coords(k) for k in ('coords_x', 'coords_y')], 1)
            element_mask = create_mask_from_coords(coords)
        else:
            element_mask = element['mask']

        # Add element to ROI mask
        ROI, element = _add_element_to_roi(
            elinfo=elinfo, ROI=ROI, GT_code=GT_code,
            element_mask=element_mask, roiinfo=roiinfo)

    except Exception as e:
        if verbose:
            print('%s: ERROR! (see below)' % monitorPrefix)
            print(e)
    return ROI, element
