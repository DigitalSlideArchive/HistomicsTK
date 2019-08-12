# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 22:50:03 2019

@author: tageldim
"""

import unittest

import os
import girder_client
from pandas import read_csv

from histomicstk.utils.polygon_and_mask_utils import (
        get_image_from_htk_response, get_bboxes_from_slide_annotations,
        _get_idxs_for_all_rois, get_roi_mask)

# %%===========================================================================
# Constants & prep work
# =============================================================================

APIURL = 'http://demo.kitware.com/histomicstk/api/v1/'
SOURCE_FOLDER_ID = '5bbdeba3e629140048d017bb'
SAMPLE_SLIDE_ID = "5bbdee92e629140048d01b5d" 
GTCODE_PATH = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 
        'test_files', 'sample_GTcodes.csv')

gc= girder_client.GirderClient(apiUrl = APIURL)
gc.authenticate(interactive=True)

# %%===========================================================================

class GirderUtilsTest(unittest.TestCase):

    def test_get_image_from_htk_response(self):
        
        getStr = "/item/%s/tiles/region?left=%d&right=%d&top=%d&bottom=%d" % (
          SAMPLE_SLIDE_ID, 59000, 59100, 35000, 35100)
        resp  = gc.get(getStr, jsonResp=False)
        rgb = get_image_from_htk_response(resp)

        self.assertTupleEqual(rgb.shape, (100, 100, 3))
        
# %%===========================================================================

class MaskUtilsTest(unittest.TestCase):

    def test_get_bboxes_from_slide_annotations(self):
        
        slide_annotations = gc.get('/annotation/item/' + SAMPLE_SLIDE_ID)
        element_infos = get_bboxes_from_slide_annotations(slide_annotations)
        
        self.assertTupleEqual(element_infos.shape, (49, 9))
        self.assertTupleEqual(
            tuple(element_infos.columns), 
            (('annidx','elementidx','type','group','xmin','xmax','ymin',
              'ymax','bbox_area')))
        
    def test_get_roi_mak(self):
        
        slide_annotations = gc.get('/annotation/item/' + SAMPLE_SLIDE_ID)
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
            verbose=False, monitorPrefix="roi 1")
        
        self.assertTupleEqual(ROI.shape, (4594, 4542))
        self.assertTupleEqual((
                roiinfo['BBOX_HEIGHT'], roiinfo['BBOX_WIDTH'],
                roiinfo['XMIN'], roiinfo['XMAX'],
                roiinfo['YMIN'], roiinfo['YMAX']), 
                (4595, 4543, 59206, 63749, 33505, 38100))
                 
        
# %%===========================================================================
        
if __name__ == '__main__':
    unittest.main()