# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 17:14:38 2019.

@author: tageldim
"""

import unittest

import os
import girder_client
from pandas import read_csv

from histomicstk.annotations_and_masks.polygon_merger import Polygon_merger
from histomicstk.annotations_and_masks.masks_to_annotations_handler import (
    get_annotation_documents_from_contours, )
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    delete_annotations_in_slide)

# %%===========================================================================
# Constants & prep work
# =============================================================================

APIURL = 'http://candygram.neurology.emory.edu:8080/api/v1/'
SAMPLE_SLIDE_ID = '5d586d76bd4404c6b1f286ae'

gc = girder_client.GirderClient(apiUrl=APIURL)
# gc.authenticate(interactive=True)
gc.authenticate(apiKey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')

# read GTCodes dataframe
PTESTS_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '..', '..',
    '..', 'plugin_tests')
GTCODE_PATH = os.path.join(PTESTS_PATH, 'test_files', 'sample_GTcodes.csv')
GTCodes_df = read_csv(GTCODE_PATH)
GTCodes_df.index = GTCodes_df.loc[:, 'group']

# This is where masks for adjacent rois are saved
MASK_LOADPATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_files', 'polygon_merger_roi_masks')
maskpaths = [
    os.path.join(MASK_LOADPATH, j) for j in os.listdir(MASK_LOADPATH)
    if j.endswith('.png')]

# %%===========================================================================


class PolygonMergerTest(unittest.TestCase):
    """Test methods for polygon merger."""

    def test_polygon_merger(self):
        """Test Polygon_merger.run()."""
        # init and run polygon merger on masks grid
        pm = Polygon_merger(
            maskpaths=maskpaths, GTCodes_df=GTCodes_df,
            discard_nonenclosed_background=True, verbose=1)
        contours_df = pm.run()

        # make sure it is what we expect
        self.assertTupleEqual(contours_df.shape, (17, 13))
        self.assertSetEqual(
            set(contours_df.loc[:, 'group']),
            {'roi', 'mostly_tumor', 'mostly_stroma',
             'mostly_lymphocytic_infiltrate'})

        # deleting existing annotations in target slide (if any)
        delete_annotations_in_slide(gc, SAMPLE_SLIDE_ID)

        # get list of annotation documents
        annotation_docs = get_annotation_documents_from_contours(
            contours_df.copy(), separate_docs_by_group=True,
            docnamePrefix='test',
            verbose=False, monitorPrefix=SAMPLE_SLIDE_ID + ": annotation docs")

        # post annotations to slide -- make sure it posts without errors
        for annotation_doc in annotation_docs:
            resp = gc.post(
                "/annotation?itemId=" + SAMPLE_SLIDE_ID, json=annotation_doc)
            self.assertTrue('annotation' in resp.keys())


# %%===========================================================================


if __name__ == '__main__':
    unittest.main()
