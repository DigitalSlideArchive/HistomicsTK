# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 17:14:38 2019.
@author: tageldim
"""

import os
import pytest

from pandas import read_csv

from htk_test_utilities import girderClient, getTestFilePath  # noqa

from histomicstk.annotations_and_masks.polygon_merger import Polygon_merger
from histomicstk.annotations_and_masks.masks_to_annotations_handler import (
    get_annotation_documents_from_contours, )
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    delete_annotations_in_slide)


class TestPolygonMerger(object):
    """Test methods for polygon merger."""

    @pytest.mark.usefixtures('girderClient')  # noqa
    def test_polygon_merger(self, girderClient):  # noqa
        """Test Polygon_merger.run()."""

        # read GTCodes dataframe
        gtcodePath = getTestFilePath('sample_GTcodes.csv')
        GTCodes_df = read_csv(gtcodePath)
        GTCodes_df.index = GTCodes_df.loc[:, 'group']

        # This is where masks for adjacent rois are saved
        MASK_LOADPATH = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'test_files', 'polygon_merger_roi_masks')
        maskpaths = [
            os.path.join(MASK_LOADPATH, j) for j in os.listdir(MASK_LOADPATH)
            if j.endswith('.png')]

        # init and run polygon merger on masks grid
        pm = Polygon_merger(
            maskpaths=maskpaths, GTCodes_df=GTCodes_df,
            discard_nonenclosed_background=True, verbose=1)
        contours_df = pm.run()

        # make sure it is what we expect
        assert contours_df.shape == (17, 13)
        assert set(contours_df.loc[:, 'group']) == {
            'roi', 'mostly_tumor', 'mostly_stroma',
            'mostly_lymphocytic_infiltrate'}

        # deleting existing annotations in target slide (if any)
        sampleSlideItem = girderClient.resourceLookup(
            '/user/admin/Public/TCGA-A2-A0YE-01Z-00-DX1.8A2E3094-5755-42BC-969D-7F0A2ECA0F39.svs')  # noqa
        sampleSlideId = str(sampleSlideItem['_id'])
        delete_annotations_in_slide(girderClient, sampleSlideId)

        # get list of annotation documents
        annotation_docs = get_annotation_documents_from_contours(
            contours_df.copy(), separate_docs_by_group=True,
            docnamePrefix='test',
            verbose=False, monitorPrefix=sampleSlideId + ": annotation docs")

        # post annotations to slide -- make sure it posts without errors
        for annotation_doc in annotation_docs:
            resp = girderClient.post(
                "/annotation?itemId=" + sampleSlideId, json=annotation_doc)
            assert 'annotation' in resp.keys()
