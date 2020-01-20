# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 22:50:03 2019.

@author: tageldim

"""
import unittest
import os
import girder_client

from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    get_image_from_htk_response, get_bboxes_from_slide_annotations,
    parse_slide_annotations_into_table)

# %%===========================================================================
# Constants & prep work
# =============================================================================

# APIURL = 'http://demo.kitware.com/histomicstk/api/v1/'
# SAMPLE_SLIDE_ID = '5bbdee92e629140048d01b5d'
APIURL = 'http://candygram.neurology.emory.edu:8080/api/v1/'
SAMPLE_SLIDE_ID = '5d586d57bd4404c6b1f28640'
GTCODE_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_files', 'sample_GTcodes.csv')

gc = girder_client.GirderClient(apiUrl=APIURL)
# gc.authenticate(interactive=True)
gc.authenticate(apiKey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')

# %%===========================================================================


class GirderUtilsTest(unittest.TestCase):
    """Test utilities for interaction with girder."""

    def test_get_image_from_htk_response(self):
        """Test get_image_from_htk_response."""
        getStr = "/item/%s/tiles/region?left=%d&right=%d&top=%d&bottom=%d" % (
            SAMPLE_SLIDE_ID, 59000, 59100, 35000, 35100)
        resp = gc.get(getStr, jsonResp=False)
        rgb = get_image_from_htk_response(resp)

        self.assertTupleEqual(rgb.shape, (100, 100, 3))

# %%===========================================================================


class MaskUtilsTest(unittest.TestCase):
    """Test utilities for makign masks."""

    def test_get_bboxes_from_slide_annotations(self):
        """Test get_bboxes_from_slide_annotations."""
        slide_annotations = gc.get('/annotation/item/' + SAMPLE_SLIDE_ID)
        element_infos = get_bboxes_from_slide_annotations(slide_annotations)

        # self.assertTupleEqual(element_infos.shape, (76, 9))
        self.assertTupleEqual(
            tuple(element_infos.columns),
            (('annidx', 'elementidx', 'type', 'group',
              'xmin', 'xmax', 'ymin', 'ymax', 'bbox_area')))

    def test_parse_slide_annotations_into_table(self):
        """Test parse_slide_annotations_into_table."""
        slide_annotations = gc.get('/annotation/item/' + SAMPLE_SLIDE_ID)
        element_infos = parse_slide_annotations_into_table(slide_annotations)

        # self.assertTupleEqual(element_infos.shape, (76, 12))
        self.assertSetEqual(
            set(element_infos.columns),
            {'annidx', 'elementidx', 'type', 'group', 'xmin', 'xmax',
             'ymin', 'ymax', 'bbox_area', 'coords_x', 'coords_y', 'color'})
        self.assertSetEqual(
            set(element_infos.loc[:, 'type']),
            {'polyline', 'rectangle', 'point'})

# %%===========================================================================


if __name__ == '__main__':
    unittest.main()
