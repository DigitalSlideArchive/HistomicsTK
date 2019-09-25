# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 01:29:00 2019.

@author: tageldim
"""

import unittest

# import os
import girder_client

from histomicstk.annotations_and_masks.polygon_merger_v2 import (
    Polygon_merger_v2, )
from histomicstk.annotations_and_masks.masks_to_annotations_handler import (
    get_annotation_documents_from_contours,
    _discard_nonenclosed_background_group, )
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    parse_slide_annotations_into_table, )
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    delete_annotations_in_slide)

# %%===========================================================================
# Constants & prep work
# =============================================================================

APIURL = 'http://candygram.neurology.emory.edu:8080/api/v1/'
SOURCE_SLIDE_ID = '5d5d6910bd4404c6b1f3d893'
POST_SLIDE_ID = '5d586d76bd4404c6b1f286ae'

gc = girder_client.GirderClient(apiUrl=APIURL)
# gc.authenticate(interactive=True)
gc.authenticate(apiKey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')

# get and parse slide annotations into dataframe
slide_annotations = gc.get('/annotation/item/' + SOURCE_SLIDE_ID)
contours_df = parse_slide_annotations_into_table(slide_annotations)

# %%===========================================================================
# Main
# =============================================================================


class PolygonMerger_v2_Test(unittest.TestCase):
    """Test methods for polygon merger v2."""

    def test_polygon_merger_v2(self):
        """Test Polygon_merger_v2.run()."""
        # init & run polygon merger
        pm = Polygon_merger_v2(contours_df, verbose=1)
        pm.unique_groups.remove("roi")
        pm.run()

        # make sure it is what we expect
        self.assertTupleEqual(pm.new_contours.shape, (16, 13))
        self.assertSetEqual(
            set(pm.new_contours.loc[:, 'group']),
            {'mostly_tumor', 'mostly_stroma', 'mostly_lymphocytic_infiltrate'})

        # add colors (aesthetic)
        for group in pm.unique_groups:
            cs = contours_df.loc[contours_df.loc[:, "group"] == group, "color"]
            pm.new_contours.loc[
                pm.new_contours.loc[:, "group"] == group, "color"] = cs.iloc[0]

        # get rid of nonenclosed stroma (aesthetic)
        pm.new_contours = _discard_nonenclosed_background_group(
            pm.new_contours, background_group="mostly_stroma")

        # deleting existing annotations in target slide (if any)
        delete_annotations_in_slide(gc, POST_SLIDE_ID)

        # get list of annotation documents
        annotation_docs = get_annotation_documents_from_contours(
            pm.new_contours.copy(), separate_docs_by_group=True,
            docnamePrefix='test',
            verbose=False, monitorPrefix=POST_SLIDE_ID + ": annotation docs")

        # post annotations to slide -- make sure it posts without errors
        for annotation_doc in annotation_docs:
            resp = gc.post(
                "/annotation?itemId=" + POST_SLIDE_ID, json=annotation_doc)
            self.assertTrue('annotation' in resp.keys())

# %%===========================================================================


if __name__ == '__main__':
    unittest.main()
