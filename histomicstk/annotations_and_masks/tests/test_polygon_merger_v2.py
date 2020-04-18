# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 01:29:00 2019.

@author: tageldim
"""

import json

import htk_test_utilities as utilities

from histomicstk.annotations_and_masks.polygon_merger_v2 import (
    Polygon_merger_v2, )
from histomicstk.annotations_and_masks.masks_to_annotations_handler import (
    get_annotation_documents_from_contours,
    _discard_nonenclosed_background_group, )
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    parse_slide_annotations_into_table, )


class TestPolygonMerger_v2(object):
    """Test methods for polygon merger v2."""

    def test_polygon_merger_v2(self):
        """Test Polygon_merger_v2.run()."""
        annotationPath = utilities.externaldata(
            'data/TCGA-A2-A0YE-01Z-00-DX1_GET_MergePolygons.svs_annotations.json.sha512')
        slide_annotations = json.load(open(annotationPath))
        contours_df = parse_slide_annotations_into_table(slide_annotations)

        # init & run polygon merger
        pm = Polygon_merger_v2(contours_df, verbose=1)
        pm.unique_groups.remove("roi")
        pm.run()

        # make sure it is what we expect
        assert pm.new_contours.shape == (16, 13)
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
