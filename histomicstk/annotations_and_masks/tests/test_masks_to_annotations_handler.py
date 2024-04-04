"""
Created on Tue Aug 20 18:13:37 2019.

@author: tageldim
"""
import os
import sys

import pandas as pd
import pytest
from imageio import imread
from pandas import read_csv

from histomicstk.annotations_and_masks.annotation_and_mask_utils import \
    delete_annotations_in_slide
from histomicstk.annotations_and_masks.masks_to_annotations_handler import (
    get_annotation_documents_from_contours, get_contours_from_mask)

thisDir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(thisDir, '../../../tests'))
from tests.htk_test_utilities import getTestFilePath, girderClient  # noqa


class TestMasksToAnnotations:
    """Test methods for getting ROI mask from annotations."""

    def _setup(self):
        # read GTCodes dataframe
        gtcodePath = getTestFilePath('sample_GTcodes.csv')
        self.GTCodes_df = read_csv(gtcodePath)
        self.GTCodes_df.index = self.GTCodes_df.loc[:, 'group']

        # read sample contours_df dataframe to test against
        contoursDfPath = getTestFilePath(os.path.join(
            'annotations_and_masks', 'sample_contours_df.tsv'))
        self.CONTOURS_DF = read_csv(contoursDfPath, sep='\t', index_col=0)

        # read mask
        self.X_OFFSET = 59206
        self.Y_OFFSET = 33505
        self.MASKNAME = (
            'TCGA-A2-A0YE-01Z-00-DX1.8A2E3094-5755-42BC-969D-7F0A2ECA0F39_'
            'left-%d_top-%d_mag-BASE.png' % (self.X_OFFSET, self.Y_OFFSET))
        MASKPATH = getTestFilePath(os.path.join(
            'annotations_and_masks', self.MASKNAME))
        self.MASK = imread(MASKPATH)

    def test_get_contours_from_mask(self):
        """Test get_contours_from_mask()."""
        self._setup()
        # get contours from mask
        # groups_to_get = [
        #     'mostly_tumor', 'mostly_stroma']
        groups_to_get = None
        contours_df = get_contours_from_mask(
            MASK=self.MASK, GTCodes_df=self.GTCodes_df,
            groups_to_get=groups_to_get,
            get_roi_contour=True, roi_group='roi',
            discard_nonenclosed_background=True,
            background_group='mostly_stroma',
            MIN_SIZE=30, MAX_SIZE=None, verbose=False,
            monitorPrefix=self.MASKNAME[:12] + ': getting contours')

        # make sure it is what we expect
        assert set(contours_df.columns) == set(self.CONTOURS_DF.columns)
        assert all(contours_df.iloc[:10, :] == self.CONTOURS_DF.iloc[:10, :])

    def test_get_contours_from_mask_with_groups(self):
        """Test get_contours_from_mask()."""
        self._setup()
        # get contours from mask
        groups_to_get = [
            'mostly_tumor', 'mostly_stroma']
        contours_df = get_contours_from_mask(
            MASK=self.MASK, GTCodes_df=self.GTCodes_df,
            groups_to_get=groups_to_get,
            get_roi_contour=True, roi_group='roi',
            discard_nonenclosed_background=True,
            background_group='mostly_stroma',
            MIN_SIZE=30, MAX_SIZE=None, verbose=False,
            monitorPrefix=self.MASKNAME[:12] + ': getting contours')

        # make sure it is what we expect
        assert set(contours_df.columns) == set(self.CONTOURS_DF.columns)
        assert all(contours_df.iloc[:10, :] == self.CONTOURS_DF.iloc[:10, :])
        assert len(contours_df) == 26

    def test_get_contours_from_mask_with_zeroes(self):
        """Test get_contours_from_mask()."""
        self._setup()
        groups_to_get = None
        gtcodes = pd.concat([self.GTCodes_df, pd.DataFrame([{
            'group': 'zeroes',
            'overlay_order': 4,
            'GT_code': 0,
            'is_roi': 0,
            'is_background_class': 0,
            'color': 'rgb(0,128,0)',
            'comments': 'zeroes'}])], ignore_index=True)
        gtcodes.index = gtcodes.loc[:, 'group']
        contours_df = get_contours_from_mask(
            MASK=self.MASK, GTCodes_df=gtcodes,
            groups_to_get=groups_to_get,
            get_roi_contour=True, roi_group='roi',
            discard_nonenclosed_background=True,
            background_group='mostly_stroma',
            MIN_SIZE=30, MAX_SIZE=None)

        # make sure it is what we expect
        assert set(contours_df.columns) == set(self.CONTOURS_DF.columns)
        assert all(contours_df.iloc[:10, :] == self.CONTOURS_DF.iloc[:10, :])
        assert len(contours_df) == 49

    @pytest.mark.usefixtures('girderClient')  # noqa
    def test_get_annotation_documents_from_contours(self, girderClient):  # noqa
        """Test get_contours_from_bin_mask()."""
        self._setup()
        original_iteminfo = girderClient.get(
            '/item', parameters={'text': 'TCGA-A2-A0YE-01Z-00-DX1'})[0]

        folder = girderClient.post(
            '/folder', data={
                'parentId': original_iteminfo['folderId'],
                'name': 'test-masks-annot-handler',
            })

        # copy the item
        sampleSlideItem = girderClient.post(
            '/item/%s/copy' % original_iteminfo['_id'], data={
                'name': 'TCGA-A2-A0YE-01Z.svs',
                'copyAnnotations': True,
                'folderId': folder['_id'],
            })
        sampleSlideId = str(sampleSlideItem['_id'])
        # get list of annotation documents
        annprops = {
            'X_OFFSET': self.X_OFFSET,
            'Y_OFFSET': self.Y_OFFSET,
            'opacity': 0.2,
            'lineWidth': 4.0,
        }
        annotation_docs = get_annotation_documents_from_contours(
            self.CONTOURS_DF.copy(), separate_docs_by_group=True,
            annots_per_doc=10, docnamePrefix='test', annprops=annprops,
            verbose=False,
            monitorPrefix=self.MASKNAME[:12] + ': annotation docs')

        # make sure its what we expect
        assert len(annotation_docs) == 8
        assert {j['name'] for j in annotation_docs} == {
            'test_blood_vessel-0',
            'test_exclude-0',
            'test_mostly_lymphocytic_infiltrate-0',
            'test_mostly_stroma-0',
            'test_mostly_tumor-0',
            'test_mostly_tumor-1',
            'test_normal_acinus_or_duct-0',
            'test_roi-0',
        }

        # deleting existing annotations in target slide (if any)
        delete_annotations_in_slide(girderClient, sampleSlideId)

        # post annotations to slide -- make sure it posts without errors
        resp = girderClient.post(
            '/annotation?itemId=' + sampleSlideId,
            json=annotation_docs[0])
        assert 'annotation' in resp.keys()
