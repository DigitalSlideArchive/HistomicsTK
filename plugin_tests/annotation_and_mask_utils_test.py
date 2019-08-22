# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 22:50:03 2019.

@author: tageldim
"""

from tests import base

from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    get_image_from_htk_response, get_bboxes_from_slide_annotations)

from .girder_client_common import GirderClientTestCase


# boiler plate to start and stop the server

def setUpModule():
    base.enabledPlugins.append('HistomicsTK')
    base.startServer(False)


def tearDownModule():
    base.stopServer()


class GirderUtilsTest(GirderClientTestCase):
    """Test utilities for interaction with girder."""

    def test_get_image_from_htk_response(self):
        """Test get_image_from_htk_response."""
        getStr = "/item/%s/tiles/region?left=%d&right=%d&top=%d&bottom=%d" % (
            self.wsiFile['itemId'], 59000, 59100, 35000, 35100)
        resp = self.gc.get(getStr, jsonResp=False)
        rgb = get_image_from_htk_response(resp)

        self.assertTupleEqual(rgb.shape, (100, 100, 3))

# %%===========================================================================


class MaskUtilsTest(GirderClientTestCase):
    """Test utilities for makign masks."""

    def test_get_bboxes_from_slide_annotations(self):
        """Test get_bboxes_from_slide_annotations."""
        slide_annotations = self.gc.get('/annotation/item/' + self.wsiFile['itemId'])
        element_infos = get_bboxes_from_slide_annotations(slide_annotations)

        self.assertTupleEqual(element_infos.shape, (76, 9))
        self.assertTupleEqual(
            tuple(element_infos.columns),
            (('annidx', 'elementidx', 'type', 'group',
              'xmin', 'xmax', 'ymin', 'ymax', 'bbox_area')))
