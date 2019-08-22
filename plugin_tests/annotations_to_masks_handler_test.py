# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 18:47:34 2019.

@author: tageldim
"""

import os
import shutil
import tempfile
from imageio import imread

from tests import base

from histomicstk.annotations_and_masks.annotations_to_masks_handler import (
    get_all_roi_masks_for_slide)

from .girder_client_common import GirderClientTestCase, GTCODE_PATH


# boiler plate to start and stop the server

def setUpModule():
    base.enabledPlugins.append('HistomicsTK')
    base.startServer(False)


def tearDownModule():
    base.stopServer()


class GetROIMasksTest(GirderClientTestCase):
    """Test methods for getting ROI mask from annotations."""

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
