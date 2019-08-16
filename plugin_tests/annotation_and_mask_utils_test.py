# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 22:50:03 2019

@author: tageldim
"""

import unittest

import os
import girder_client

from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    get_image_from_htk_response, get_bboxes_from_slide_annotations)

# %%===========================================================================
# Constants & prep work
# =============================================================================

APIURL = 'http://demo.kitware.com/histomicstk/api/v1/'
SOURCE_FOLDER_ID = '5bbdeba3e629140048d017bb'
SAMPLE_SLIDE_ID = '5bbdee92e629140048d01b5d'
GTCODE_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_files', 'sample_GTcodes.csv')

gc = girder_client.GirderClient(apiUrl=APIURL)
gc.authenticate(interactive=True)

# %%===========================================================================


class GiirderUtilsTest(unittest.TestCase):

    def test_get_image_from_htk_response(self):

        getStr = "/item/%s/tiles/region?left=%d&right=%d&top=%d&bottom=%d" % (
            SAMPLE_SLIDE_ID, 59000, 59100, 35000, 35100)
        resp = gc.get(getStr, jsonResp=False)
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
            (('annidx', 'elementidx', 'type', 'group',
              'xmin', 'xmax', 'ymin', 'ymax', 'bbox_area')))

# %%===========================================================================


if __name__ == '__main__':
    unittest.main()
