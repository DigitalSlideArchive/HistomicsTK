# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 18:47:34 2019

@author: tageldim
"""

import unittest

import os
import girder_client
from pandas import read_csv
from imageio import imread

from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
        get_bboxes_from_slide_annotations, _get_idxs_for_all_rois)
from histomicstk.annotations_and_masks.annotations_to_masks_handler import (
        get_roi_mask, get_all_roi_masks_for_slide)

# %%===========================================================================
# Constants & prep work
# =============================================================================

APIURL = 'http://demo.kitware.com/histomicstk/api/v1/'
#SOURCE_FOLDER_ID = '5bbdeba3e629140048d017bb'
SAMPLE_SLIDE_ID = "5bbdee92e629140048d01b5d" 
GTCODE_PATH = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 
        'test_files', 'sample_GTcodes.csv')
MASK_SAVEPATH = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '../../')


gc= girder_client.GirderClient(apiUrl = APIURL)
gc.authenticate(interactive=True)

# %%===========================================================================

class GetROIMasksTest(unittest.TestCase):
        
    def test_get_roi_mask(self):
        
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
        
    # %% ----------------------------------------------------------------------
    
    def test_get_all_roi_masks_for_slide(self):
        
        get_all_roi_masks_for_slide(
            gc=gc, slide_id=SAMPLE_SLIDE_ID, GTCODE_PATH=GTCODE_PATH, 
            MASK_SAVEPATH=MASK_SAVEPATH, 
            get_roi_mask_kwargs = {
                'iou_thresh': 0.0, 'crop_to_roi': True, 'verbose': True},
            )
        
        left = 59206
        top = 33505
        expected_savename = 'TCGA-A2-A0YE-01Z-00-DX1.8A2E3094-5755-42BC-969D'
        expected_savename += '-7F0A2ECA0F39_left-%d_top-%d_mag-BASE.png' % (
                left, top)
        ROI = imread(os.path.join(MASK_SAVEPATH, expected_savename))
        
        self.assertTupleEqual(ROI.shape, (4594, 4542))
        
                 
# %%===========================================================================
        
if __name__ == '__main__':
    unittest.main()