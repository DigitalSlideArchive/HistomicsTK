# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 22:30:06 2019

@author: tageldim
"""

from io import BytesIO
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = 1000000000
import numpy as np
from pandas import DataFrame, read_csv

#%% ===========================================================================

def get_image_from_htk_response(resp):
    """Given a histomicsTK girder response, get np array image.
    
    Arguments:
        * resp - response from server request
    
    Returns:
        * a pillow Image object
        
    Example:
        gc= girder_client.GirderClient(apiUrl = APIURL)
        gc.authenticate(interactive=True)
        getStr = "/item/%s/tiles/region?left=%d&right=%d&top=%d&bottom=%d" % (
          slide_id, left, right, top, bottom)
        resp  = gc.get(getStr, jsonResp=False)
        rgb = get_image_from_htk_response(resp)
    """        
    image_content = BytesIO(resp.content)
    image_content.seek(0)
    Image.open(image_content)
    image = Image.open(image_content)
    return np.uint8(image)

#%% ===========================================================================
    
def rotate_point_list(point_list, rotation, center=(0, 0)):  
    """Rotates a certain point list around a central point. 
    Modified from javascript version at:
        https://github.com/girder/large_image/blob/master/ ...
        web_client/annotations/rotate.js
    
    Arguments:
        * point_list - list of tuples (x, y)
        * rotation - degrees (in radians) 
        * center - central point coordinates
    
    Returns:
        * list of tuples 
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

#%%============================================================================

def get_rotated_rectangular_coords(
        roi_center, roi_width, roi_height, roi_rotation=0):
    """Given data on rectangular ROI center/width/height/rotation, 
    get the unrotated abounding box coordinates around rotated ROI. This 
    of course is applicable to any rotated rectangular annotation.
    
    Arguments:
        * roi_center - [x, y] format
        * roi_width, roi_height, roi_rotation - float
    
    Returns:
        * dict with roi corners (x, y) and bounds
    """
    # Get false bounds
    x_min = roi_center[0] - int(roi_width/2)
    x_max = roi_center[0] + int(roi_width/2)
    y_min = roi_center[1] - int(roi_height/2)
    y_max = roi_center[1] + int(roi_height/2)
    
    # Get pseudo-corners of rectangle (before rotation)
    roi_corners_false = [(x_min, y_min), (x_max, y_min), 
                         (x_max, y_max), (x_min, y_max)]
    
    # Get *true* coordinates of rectangle corners 
    roi_corners = rotate_point_list(roi_corners_false, 
                                    rotation=roi_rotation, 
                                    center=roi_center)
    roi_corners = np.array(roi_corners)
    
    # pack into dict
    roi_info = {
        'roi_corners' : roi_corners, 
        'x_min' : roi_corners[:, 0].min(), 
        'x_max' : roi_corners[:, 0].max(), 
        'y_min' : roi_corners[:, 1].min(), 
        'y_max' : roi_corners[:, 1].max()
    }
    
    return roi_info

#%% ===========================================================================

def get_bboxes_from_slide_annotations(slide_annotations):
    """Given a slide annotation list, gets information on bounding boxes.
    
    Arguments:
        * slide_annotations - response from server request (list of dicts)
    
    Returns:
        * A pandas DataFrame. The columns annidx and elementidx encode the
          dict index of annotation document and element, respectively, in the 
          original slide_annotations list of dictionaries
        
    Example:
        gc= girder_client.GirderClient(apiUrl = APIURL)
        gc.authenticate(interactive=True)
        slide_annotations = gc.get('/annotation/item/' + SAMPLE_SLIDE_ID)
        element_infos  = get_bboxes_from_slide_annotations(slide_annotations)
    """
    element_infos = DataFrame(columns=[
            'annidx','elementidx', 'type', 'group', 
            'xmin', 'xmax', 'ymin', 'ymax',])
        
    for annidx, ann in enumerate(slide_annotations):
        for elementidx, element in enumerate(ann['annotation']['elements']):
            
            elno = element_infos.shape[0]
            element_infos.loc[elno, 'annidx'] = annidx
            element_infos.loc[elno, 'elementidx'] = elementidx
            element_infos.loc[elno, 'type'] = element['type']
            
            # get bounds
            if element['type'] == 'polyline':
                coords = np.array(element['points'])[:, :-1]
                xmin, ymin = [int(j) for j in np.min(coords, axis=0)]
                xmax, ymax = [int(j) for j in np.max(coords, axis=0)]
            
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
                    (ymax-ymin) * (xmax-xmin))    
    
    return element_infos

#%% ===========================================================================

def np_vec_no_jit_iou(bboxes1, bboxes2):
    """Fast, vectorized IoU.
    Source: https://medium.com/@venuktan/vectorized-intersection-over-union ...
            -iou-in-numpy-and-tensor-flow-4fa16231b63d
    
    Arguments:
        * bboxes1 - np array where columns encode bounding box 
                  corners xmin, ymin, xmax, ymax
        * bboxes2 - same as bboxes 1
    
    Returns:
        * np array of IoU values for each pair from bboxes1 & bboxes2
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

#%% ===========================================================================

def _get_idxs_for_all_rois(GTCodes, element_infos):
    """Gets indices of ROIs within the element_infos dataframe. (Internal)"""    
    roi_labels = list(GTCodes.loc[GTCodes.loc[:, 'is_roi'] == 1, 'group'])
    idxs_for_all_rois = []
    for idx, elinfo in element_infos.iterrows():
        if elinfo.group in roi_labels:
            idxs_for_all_rois.append(idx)
    return idxs_for_all_rois

#%% ===========================================================================

def get_idxs_for_annots_overlapping_roi_by_bbox(
        element_infos, idx_for_roi, iou_thresh=0.0):
    """Find indices of **potentially** included annoations within the ROI
    We say "potentially" because this uses the IoU of roi and annotation 
    as a fast indicator of potential inclusion. This helps dramatically scale 
    down the number of annotations to look through. Later on, a detailed look
    at wheter the annotation polygons actually overlap the ROI can be done.
    
    Arguments:
        * element_infos - pandas DataFrame 
               from running get_bboxes_from_slide_annotations()
        * idx_for_roi - index for roi annotation within the element_infos DF
        * iou_thresh - overlap threshold to be considered within ROI
    
    Returns:
        * list of indices relative to element_infos
    """
    bboxes = np.array(
            element_infos.loc[:, ['xmin', 'ymin', 'xmax', 'ymax']], 
            dtype='int')
    iou = np_vec_no_jit_iou(bboxes[idx_for_roi, :][None,...], bboxes2=bboxes)
    iou = np.concatenate((np.arange(iou.shape[1])[None, ...], iou))
    iou = iou[:, iou[1, :] > iou_thresh]    
    
    overlaps = set([int(j) for j in iou[0, :]]) - {idx_for_roi}
    
    return list(overlaps)

#%% ===========================================================================

def create_mask_from_coords(coords):
    """ Creates a binary mask from given vertices coordinates. 
    Source: This is modified from code by Juan Carlos from David Gutman Lab.    
    
    Arguments:
        * vertices - must be in numpy ndarray form 
          (e.g. ([x1,y1],[x2,y2],[x3,y3],.....,[xn,yn])), 
          where xn and yn corresponds to the nth vertix coordinate. 
    
    Returns:
        * mask: a numpy array binary mask
    """
    
    polygon = coords.copy()
    
    # use the smallest bounding region, calculated from vertices
    min_x, min_y = np.min(polygon, axis=0)
    max_x, max_y = np.max(polygon, axis=0)
    # get the new width and height
    width = int(max_x - min_x)
    height = int(max_y - min_y)
    # shift all vertices to account for location of the smallest bounding box
    polygon[:,0] = polygon[:,0] - min_x
    polygon[:,1] = polygon[:,1] - min_y
    
    # convert to tuple form for masking function (nd array does not work)
    vertices_tuple = tuple(map(tuple, polygon))
    # make the mask
    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon(vertices_tuple, outline=1, fill=1)
    
    return np.array(img, dtype='int32')

#%% ===========================================================================

def _get_element_mask(elinfo, slide_annotations):
    """get coordinates and mask for annotation element. (Internal)"""
    
    element = slide_annotations[int(elinfo['annidx'])][
            'annotation']['elements'][int(elinfo['elementidx'])]
    
    if elinfo['type'] == 'polyline':
        coords = np.array(element['points'])[:, :-1]
    elif elinfo['type'] == 'rectangle':
        infoDict = get_rotated_rectangular_coords(
            roi_center=element['center'], roi_width=element['width'], 
            roi_height=element['height'], roi_rotation=element['rotation'])
        coords = infoDict['roi_corners']
    
    mask = create_mask_from_coords(coords)
    return coords, mask

#%% ===========================================================================

def _add_element_to_roi(elinfo, ROI, GT_code, element_mask, roiinfo):
    """add single polygon to ROI given mask (Internal)"""
    
    ymin = int(elinfo.ymin - roiinfo['YMIN'])
    ymax = int(elinfo.ymax - roiinfo['YMIN'])
    xmin = int(elinfo.xmin - roiinfo['XMIN'])
    xmax = int(elinfo.xmax - roiinfo['XMIN'])
    patch = ROI[ymin:ymax, xmin:xmax]
    
    # to account for non-integer pixels
    if element_mask.shape != patch.shape:
        element_mask = np.pad(element_mask, pad_width=(
            (patch.shape[0] - element_mask.shape[0],0),
            (patch.shape[1] - element_mask.shape[1],0)), mode='constant')
    
    # add to ROI mask 
    patch[element_mask > 0] = GT_code 
    ROI[ymin:ymax, xmin:xmax] = patch.copy()
    
    return ROI

#%% ===========================================================================

def get_roi_mask(
        slide_annotations, element_infos, GTCodes_df, 
        idx_for_roi, iou_thresh=0.0, roiinfo=None,
        crop_to_roi=True, verbose=False, monitorPrefix=""):
    """Parses annotations and gets a ground truth mask for a single 
    region of interest (ROI). This will look at all slide annotations and 
    get ones that overlap with the ROI and assigns them into a mask.
    
    Arguments:
        * slide_annotations - response from server request (list of dicts)
          eg. slide_annotations = gc.get('/annotation/item/' + SLIDE_ID)
        * element_infos - A pandas DataFrame. The columns annidx and elementidx 
          encode the dict index of annotation document and element, 
          respectively, in the original slide_annotations list of dictionaries.
          This can be obain by get_bboxes_from_slide_annotations() method, eg.
          element_infos  = get_bboxes_from_slide_annotations(slide_annotations)
        * GTCodes_df - the ground truth codes and information dataframe. 
          WARNING: Modified indide this method so pass a copy.
          This is a dataframe that is indexed by the annotation group name and
          has the following columns:
              - group: group name of annotation (string), eg. "mostly_tumor"
              - overlay_order: int, how early to place the annotation in the 
                 mask. Larger values means this annotation group is overlayed
                 last and overwrites whatever overlaps it.
              - GT_code: int, desired ground truth code (in the mask)
                 Pixels of this value belong to corresponding group (class)
              - is_roi: Flag for whether this group encodes an ROI
              - is_background_class: Flag, whether this group is the default 
                fill value inside the ROI. For example, you may descide that 
                any pixel inside the ROI is considered stroma.
        * idx_for_roi - index of ROI within the element_infos dataframe.
        * iou_thresh - float, how much bounding box overlap is enough to 
           consider an annotation to belong to the region of interest
        * roiinfo (optional) - pandas series or dict containing information
           about the roi. Keys will be added to this index containing info
           about the roi like bounding box location and size.
         * crop_to_roi - flag of whether to crop polygons to roi 
           (prevent 'overflow' beyond roi edge)
         * verbose (optional) - flag. Print progress to screen?
         * monitorPrefix (optional) - string, to prepend to printed statements
    
    Returns:
        * Np array (N x 2), where pixel values encode class membership  
        * Dict of information about ROI 
        
    Example:
        gc= girder_client.GirderClient(apiUrl = APIURL)
        gc.authenticate(interactive=True)
            
        # get annotations for slide
        slide_annotations = gc.get('/annotation/item/' + SAMPLE_SLIDE_ID)
            
        # get bounding box information for all annotations
        element_infos = get_bboxes_from_slide_annotations(slide_annotations)
        
        # read ground truth codes and information
        GTCodes = read_csv(GTCODE_PATH)
        GTCodes.index = GTCodes.loc[:, 'group']
        
        # get indices of rois
        idxs_for_all_rois = _get_idxs_for_all_rois(
                GTCodes=GTCodes, element_infos=element_infos)
        
        # get roi mask and info
        ROI, roiinfo = get_roi_mask(
            slide_annotations=slide_annotations, element_infos=element_infos, 
            GTCodes_df=GTCodes.copy(), 
            idx_for_roi = idxs_for_all_rois[0], # <- let's focus on first ROI, 
            iou_thresh=0.0, roiinfo=None, crop_to_roi=True, 
            verbose=True, monitorPrefix="roi 1")
    """

    # This stores information about the ROI like bounds, slide_name, etc
    # Allows passing many parameters and good forward/backward compatibility
    if roiinfo is None:
        roiinfo = dict()
    
    # isolate annotations that potentially overlap (belong to) mask (incl. ROI)
    overlaps = get_idxs_for_annots_overlapping_roi_by_bbox(
                element_infos, idx_for_roi=idx_for_roi, iou_thresh=iou_thresh)
    elinfos_roi = element_infos.loc[[idx_for_roi,] + overlaps, :]
    
    # Add roiinfo
    roiinfo['XMIN'] = int(np.min(elinfos_roi.xmin))
    roiinfo['YMIN'] = int(np.min(elinfos_roi.ymin))
    roiinfo['XMAX'] = int(np.max(elinfos_roi.xmax))
    roiinfo['YMAX'] = int(np.max(elinfos_roi.ymax))
    roiinfo['BBOX_WIDTH'] = roiinfo['XMAX'] - roiinfo['XMIN']
    roiinfo['BBOX_HEIGHT'] = roiinfo['YMAX'] - roiinfo['YMIN']
    
    # Init mask
    ROI = np.zeros(
            (roiinfo['BBOX_HEIGHT'], roiinfo['BBOX_WIDTH']), dtype=np.uint8)
    
    # make sure ROI is overlayed first & is assigned background class if relevant
    roi_group = elinfos_roi.loc[idx_for_roi, 'group']
    GTCodes_df.loc[roi_group, 'overlay_order'] = np.min(
            GTCodes_df.loc[:, 'overlay_order']) - 1
    bck_classes = GTCodes_df.loc[
            GTCodes_df.loc[:, 'is_background_class'] == 1, :]
    if bck_classes.shape[0] > 0:
        GTCodes_df.loc[
            roi_group, 'GT_code'] = bck_classes.iloc[0, :]['GT_code']
    
    # Add annotations in overlay order
    overlay_orders = list(set(GTCodes_df.loc[:, 'overlay_order']))
    overlay_orders.sort()
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
            if verbose: print(elcountStr)
            
            try:
                # Add element to ROI mask
                _, element_mask = _get_element_mask(
                        elinfo=elinfo, slide_annotations=slide_annotations)
                ROI = _add_element_to_roi(
                    elinfo=elinfo, ROI=ROI, 
                    GT_code=GTCodes_df.loc[elinfo['group'], 'GT_code'], 
                    element_mask=element_mask, roiinfo=roiinfo)
            except Exception as e:
                if verbose: 
                    print("%s: ERROR! (see below)" % elcountStr)
                    print(e)
            
            # save a copy of ROI-only mask to crop to it later if needed
            if crop_to_roi and (
                    overlay_level == GTCodes_df.loc[roi_group, 'overlay_order']):
                roi_only_mask = ROI.copy()
        
    # Now crop polygons to roi if needed (prevent 'overflow' beyond roi edge)
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

#%% ===========================================================================

    

#%% ===========================================================================
#%% ===========================================================================

##import os    
#import girder_client 
#
#APIURL = 'http://demo.kitware.com/histomicstk/api/v1/'
#SOURCE_FOLDER_ID = '5bbdeba3e629140048d017bb'
#SAMPLE_SLIDE_ID = "5bbdee92e629140048d01b5d"
##GTCODE_PATH = os.path.join(
##        os.path.dirname(os.path.realpath(__file__)), 
##        'test_files', 'sample_GTcodes.csv')
#GTCODE_PATH = 'C:\\Users\\tageldim\\Desktop\\HistomicsTK\\plugin_tests\\test_files\\sample_GTcodes.csv'
#
##%% ---------------------------------------------------------------------
#
#gc= girder_client.GirderClient(apiUrl = APIURL)
#gc.authenticate(interactive=True)
#    
## get annotations for slide
#slide_annotations = gc.get('/annotation/item/' + SAMPLE_SLIDE_ID)
#    
## get bounding box information for all annotations
#element_infos = get_bboxes_from_slide_annotations(slide_annotations)
#
##%% ---------------------------------------------------------------------
#
## read ground truth codes and information
#GTCodes = read_csv(GTCODE_PATH)
#GTCodes.index = GTCodes.loc[:, 'group']
#
## get indices of rois
#idxs_for_all_rois = _get_idxs_for_all_rois(
#        GTCodes=GTCodes, element_infos=element_infos)
#
## get roi mask and info
#ROI, roiinfo = get_roi_mask(
#    slide_annotations=slide_annotations, element_infos=element_infos, 
#    GTCodes_df=GTCodes.copy(), 
#    idx_for_roi = idxs_for_all_rois[0], # <- let's focus on first ROI, 
#    iou_thresh=0.0, roiinfo=None, crop_to_roi=True, 
#    verbose=True, monitorPrefix="roi 1")
#
#
##%% ---------------------------------------------------------------------      
