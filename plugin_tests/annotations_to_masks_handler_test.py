# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 18:47:34 2019.

@author: tageldim
"""

import os
import shutil
import tempfile
from pandas import read_csv
from imageio import imread

from tests import base

from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    get_bboxes_from_slide_annotations, _get_idxs_for_all_rois)
from histomicstk.annotations_and_masks.annotations_to_masks_handler import (
    get_roi_mask, get_all_roi_masks_for_slide)

from .girder_client_common import GirderClientTestCase, GTCODE_PATH


# boiler plate to start and stop the server

def setUpModule():
    base.enabledPlugins.append('HistomicsTK')
    base.startServer(False)


def tearDownModule():
    base.stopServer()


# %%===========================================================================
# Constants & prep work
# =============================================================================


class GetROIMasksTest(GirderClientTestCase):
    """Test methods for getting ROI mask from annotations."""

    def test_get_roi_mask(self):
        """Test get_roi_mask()."""
        slide_annotations = self.gc.get('/annotation/item/' + self.wsiFile['itemId'])
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
            idx_for_roi=idxs_for_all_rois[0],  # <- let's focus on first ROI,
            iou_thresh=0.0, roiinfo=None, crop_to_roi=True, use_shapely=True,
            verbose=False, monitorPrefix="roi 1")

        self.assertTupleEqual(ROI.shape, (4594, 4542))
        self.assertTupleEqual((
            roiinfo['BBOX_HEIGHT'], roiinfo['BBOX_WIDTH'],
            roiinfo['XMIN'], roiinfo['XMAX'],
            roiinfo['YMIN'], roiinfo['YMAX']),
            (4595, 4543, 59206, 63749, 33505, 38100))

    # %% ----------------------------------------------------------------------

    def test_get_all_roi_masks_for_slide(self):
        """Test get_all_roi_masks_for_slide()."""
        mask_savepath = tempfile.mkdtemp()
        get_all_roi_masks_for_slide(
            gc=self.gc, slide_id=self.wsiFile['itemId'], GTCODE_PATH=GTCODE_PATH,
            MASK_SAVEPATH=mask_savepath,
            get_roi_mask_kwargs={
                'iou_thresh': 0.0, 'crop_to_roi': True, 'use_shapely': True,
                'verbose': True},
        )

        left = 59206
        top = 33505
        expected_savename = 'TCGA-A2-A0YE-01Z-00-DX1.8A2E3094-5755-42BC-969D'
        expected_savename += '-7F0A2ECA0F39_left-%d_top-%d_mag-BASE.png' % (
            left, top)
        ROI = imread(os.path.join(mask_savepath, expected_savename))

        self.assertTupleEqual(ROI.shape, (4594, 4542))
        shutil.rmtree(mask_savepath)
