# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 18:13:37 2019.

@author: tageldim
"""

import unittest

import os
import girder_client
from pandas import read_csv
from imageio import imread

from histomicstk.annotations_and_masks.masks_to_annotations_handler import (
    get_contours_from_mask, get_annotation_documents_from_contours)
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    delete_annotations_in_slide)

# %%===========================================================================
# Constants & prep work
# =============================================================================

# APIURL = 'http://demo.kitware.com/histomicstk/api/v1/'
# SAMPLE_SLIDE_ID = '5bbdee92e629140048d01b5d'
APIURL = 'http://candygram.neurology.emory.edu:8080/api/v1/'
SAMPLE_SLIDE_ID = '5d586d76bd4404c6b1f286ae'

gc = girder_client.GirderClient(apiUrl=APIURL)
# gc.authenticate(interactive=True)
gc.authenticate(apiKey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')

# read GTCodes dataframe
GTCODE_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_files', 'sample_GTcodes.csv')
GTCodes_df = read_csv(GTCODE_PATH)
GTCodes_df.index = GTCodes_df.loc[:, 'group']

# read sample contours_df dataframe to test against
CONTOURS_DF_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_files', 'sample_contours_df.tsv')
CONTOURS_DF = read_csv(CONTOURS_DF_PATH, sep='\t', index_col=0)

# read mask
X_OFFSET = 59206
Y_OFFSET = 33505
MASKNAME = "TCGA-A2-A0YE-01Z-00-DX1.8A2E3094-5755-42BC-969D-7F0A2ECA0F39" + \
    "_left-%d_top-%d_mag-BASE.png" % (X_OFFSET, Y_OFFSET)
MASKPATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_files', MASKNAME)
MASK = imread(MASKPATH)

# %%===========================================================================


class MasksToAnnotationsTest(unittest.TestCase):
    """Test methods for getting ROI mask from annotations."""

    def test_get_contours_from_mask(self):
        """Test get_contours_from_mask()."""
        # get contours from mask
        # groups_to_get = [
        #     'mostly_tumor', 'mostly_stroma']
        groups_to_get = None
        contours_df = get_contours_from_mask(
            MASK=MASK, GTCodes_df=GTCodes_df, groups_to_get=groups_to_get,
            get_roi_contour=True, roi_group='roi',
            discard_nonenclosed_background=True,
            background_group='mostly_stroma',
            MIN_SIZE=30, MAX_SIZE=None, verbose=False,
            monitorPrefix=MASKNAME[:12] + ": getting contours")

        # make sure it is what we expect
        # self.assertTupleEqual(contours_df.shape, CONTOURS_DF.shape)
        self.assertSetEqual(
            set(contours_df.columns), set(CONTOURS_DF.columns))
        self.assertTrue(all(
            contours_df.iloc[:10, :] == CONTOURS_DF.iloc[:10, :]))

    # %% ----------------------------------------------------------------------

    def test_get_annotation_documents_from_contours(self):
        """Test get_contours_from_bin_mask()."""
        # get list of annotation documents
        annprops = {
            'X_OFFSET': X_OFFSET,
            'Y_OFFSET': Y_OFFSET,
            'opacity': 0.2,
            'lineWidth': 4.0,
        }
        annotation_docs = get_annotation_documents_from_contours(
            CONTOURS_DF.copy(), separate_docs_by_group=True, annots_per_doc=10,
            docnamePrefix='test', annprops=annprops,
            verbose=False, monitorPrefix=MASKNAME[:12] + ": annotation docs")

        # make sure its what we expect
        self.assertTrue(len(annotation_docs) == 8)
        self.assertSetEqual(
            {j['name'] for j in annotation_docs},
            {
                'test_blood_vessel-0',
                'test_exclude-0',
                'test_mostly_lymphocytic_infiltrate-0',
                'test_mostly_stroma-0',
                'test_mostly_tumor-0',
                'test_mostly_tumor-1',
                'test_normal_acinus_or_duct-0',
                'test_roi-0'
            }
        )

        # deleting existing annotations in target slide (if any)
        delete_annotations_in_slide(gc, SAMPLE_SLIDE_ID)

        # post annotations to slide -- make sure it posts without errors
        resp = gc.post(
            "/annotation?itemId=" + SAMPLE_SLIDE_ID,
            json=annotation_docs[0])
        self.assertTrue('annotation' in resp.keys())

# %%===========================================================================


if __name__ == '__main__':
    unittest.main()
