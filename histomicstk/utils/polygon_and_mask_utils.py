# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 22:30:06 2019

@author: tageldim
"""

from io import BytesIO
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
import numpy as np
from pandas import DataFrame

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
    """
    Rotates a certain point list around a central point. 
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

def get_rectangular_roi_coords(
        roi_center, roi_width, roi_height, roi_rotation=0):
    """ Given data on rectangular ROI center/width/height/rotation, 
    get the unrotated abounding box coordinates around rotated ROI    
    Arguments:
        roi_center - [x, y] format
        roi_width, roi_height, roi_rotation - float
    Returns:
        dict with roi corners and bounds
    """
    # Get false bounds
    x_min = roi_center[0] - int(roi_width/2)
    x_max = roi_center[0] + int(roi_width/2)
    y_min = roi_center[1] - int(roi_height/2)
    y_max = roi_center[1] + int(roi_height/2)
    
    # Get pseudo-corners of rectangle (before rotation)
    roi_corners_false = [(x_min, y_min), (x_max, y_min), 
                         (x_max, y_max), (x_min, y_max)]
    
    # Get true coordinates of rectangle corners 
    roi_corners = rotate_point_list(roi_corners_false, 
                                    rotation=roi_rotation, 
                                    center=roi_center)
    roi_corners = np.array(roi_corners)
    
    # pack into dict
    roi_info = {'x_min' : roi_corners[:, 0].min(), 
                'x_max' : roi_corners[:, 0].max(), 
                'y_min' : roi_corners[:, 1].min(), 
                'y_max' : roi_corners[:, 1].max()}
    
    return roi_info

#%% ===========================================================================

def get_bboxes_from_slide_annotations(slide_annotations):
    """Given a slide annotation list, gets information on bounding boxes
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
                roiinfo = get_rectangular_roi_coords(
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
    
#import girder_client 
#APIURL = 'http://demo.kitware.com/histomicstk/api/v1/'
#SOURCE_FOLDER_ID = '5bbdeba3e629140048d017bb'
#SAMPLE_SLIDE_ID = "5bbdeed1e629140048d01bcb"
#
#gc= girder_client.GirderClient(apiUrl = APIURL)
#gc.authenticate(interactive=True)
#    
## get annotations for slide
#slide_annotations = gc.get('/annotation/item/' + SAMPLE_SLIDE_ID)
#    
##%%
#
#element_infos = get_bboxes_from_slide_annotations(slide_annotations)
#

















