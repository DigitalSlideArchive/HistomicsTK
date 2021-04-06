"""
Created on Fri Aug 23 17:14:38 2019.
@author: tageldim
"""
import json
import os
import pytest
from pandas import read_csv
from histomicstk.annotations_and_masks.polygon_merger import Polygon_merger
from histomicstk.annotations_and_masks.polygon_merger_v2 import \
    Polygon_merger_v2
from histomicstk.annotations_and_masks.masks_to_annotations_handler import (
    get_annotation_documents_from_contours,
    _discard_nonenclosed_background_group, )
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    parse_slide_annotations_into_tables, delete_annotations_in_slide)
import sys
thisDir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(thisDir, '../../../'))
from tests.datastore import datastore  # noqa
from tests.htk_test_utilities import girderClient, getTestFilePath  # noqa


class TestPolygonMerger:
    """Test methods for polygon merger."""

    @pytest.mark.usefixtures('girderClient')  # noqa
    def test_polygon_merger_tiled_masks(self, girderClient):  # noqa
        """Test Polygon_merger.run()."""
        # read GTCodes dataframe
        gtcodePath = getTestFilePath('sample_GTcodes.csv')
        GTCodes_df = read_csv(gtcodePath)
        GTCodes_df.index = GTCodes_df.loc[:, 'group']

        # This is where masks for adjacent rois are saved
        mask_loadpath = getTestFilePath(os.path.join(
            'annotations_and_masks', 'polygon_merger_roi_masks'))
        maskpaths = [
            os.path.join(mask_loadpath, j) for j in os.listdir(mask_loadpath)
            if j.endswith('.png')]

        # init and run polygon merger on masks grid
        pm = Polygon_merger(
            maskpaths=maskpaths, GTCodes_df=GTCodes_df,
            discard_nonenclosed_background=True, verbose=1)
        contours_df = pm.run()

        # make sure it is what we expect
        assert contours_df.shape == (13, 13)
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

    def test_polygon_merger_rtree(self):
        """Test Polygon_merger_v2.run()."""
        annotationPath = datastore.fetch(
            'TCGA-A2-A0YE-01Z-00-DX1_GET_MergePolygons.svs_annotations.json')  # noqa
        slide_annotations = json.load(open(annotationPath))
        _, contours_df = parse_slide_annotations_into_tables(slide_annotations)

        # init & run polygon merger
        pm = Polygon_merger_v2(contours_df, verbose=0)
        pm.unique_groups.remove("roi")
        pm.run()

        # make sure it is what we expect
        assert pm.new_contours.shape == (16, 16)
        assert set(pm.new_contours.loc[:, 'group']) == {
            'mostly_tumor', 'mostly_stroma', 'mostly_lymphocytic_infiltrate'}

        # add colors (aesthetic)
        for group in pm.unique_groups:
            cs = contours_df.loc[contours_df.loc[:, "group"] == group, "color"]
            pm.new_contours.loc[
                pm.new_contours.loc[:, "group"] == group, "color"] = cs.iloc[0]

        # get rid of nonenclosed stroma (aesthetic)
        pm.new_contours = _discard_nonenclosed_background_group(
            pm.new_contours, background_group="mostly_stroma")

        # get list of annotation documents
        annotation_docs = get_annotation_documents_from_contours(
            pm.new_contours.copy(), separate_docs_by_group=True,
            docnamePrefix='test',
            verbose=False, monitorPrefix="annotation docs")
        assert len(annotation_docs) == 3
