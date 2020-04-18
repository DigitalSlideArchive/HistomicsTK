# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 18:47:34 2019.

@author: tageldim

"""
import json
import os
import pytest
from pandas import read_csv
from imageio import imread
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    get_bboxes_from_slide_annotations, _get_idxs_for_all_rois)
from histomicstk.annotations_and_masks.annotations_to_masks_handler import (
    get_roi_mask, get_all_roi_masks_for_slide)

import sys
thisDir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(thisDir, '../../../tests'))

import htk_test_utilities as utilities  # noqa
from htk_test_utilities import girderClient  # noqa


class TestGetROIMasks(object):
    """Test methods for getting ROI mask from annotations."""

    def test_get_roi_mask(self):
        """Test get_roi_mask()."""
        annotationPath = utilities.externaldata(
            'data/TCGA-A2-A0YE-01Z-00-DX1.8A2E3094-5755-42BC-969D-'
            '7F0A2ECA0F39.svs_annotations.json.sha512')
        slide_annotations = json.load(open(annotationPath))
        element_infos = get_bboxes_from_slide_annotations(slide_annotations)

        # read ground truth codes and information
        testDir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), '..', '..', '..', 'tests')
        gtcodePath = os.path.join(testDir, 'test_files', 'sample_GTcodes.csv')
        GTCodes = read_csv(gtcodePath)
        GTCodes.index = GTCodes.loc[:, 'group']

        # get indices of rois
        idxs_for_all_rois = _get_idxs_for_all_rois(
            GTCodes=GTCodes, element_infos=element_infos)

        # get roi mask and info
        ROI, roiinfo = get_roi_mask(
            slide_annotations=slide_annotations, element_infos=element_infos,
            GTCodes_df=GTCodes.copy(),
            idx_for_roi=idxs_for_all_rois[0],  # <- let's focus on first ROI,
            iou_thresh=0.0, roiinfo=None, crop_to_roi=True, use_shapely=True,
            verbose=False, monitorPrefix="roi 1")

        assert ROI.shape == (4594, 4542)
        assert (
            roiinfo['BBOX_HEIGHT'], roiinfo['BBOX_WIDTH'],
            roiinfo['XMIN'], roiinfo['XMAX'],
            roiinfo['YMIN'], roiinfo['YMAX']
        ) in [
            (4820, 7006, 59206, 66212, 33505, 38325),
            (4595, 4543, 59206, 63749, 33505, 38100),
        ]

    # %% ----------------------------------------------------------------------

    @pytest.mark.usefixtures('girderClient')  # noqa
    def test_get_all_roi_masks_for_slide(self, tmpdir, girderClient):  # noqa
        """Test get_all_roi_masks_for_slide()."""
        # just a temp directory to save masks for now
        mask_savepath = str(tmpdir)

        testDir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), '..', '..', '..', 'tests')
        gtcodePath = os.path.join(testDir, 'test_files', 'sample_GTcodes.csv')
        sampleSlideItem = girderClient.resourceLookup(
            '/user/admin/Public/TCGA-A2-A0YE-01Z-00-DX1.8A2E3094-5755-42BC-969D-7F0A2ECA0F39.svs')
        sampleSlideId = str(sampleSlideItem['_id'])

        get_all_roi_masks_for_slide(
            gc=girderClient, slide_id=sampleSlideId, GTCODE_PATH=gtcodePath,
            MASK_SAVEPATH=mask_savepath,
            get_roi_mask_kwargs={
                'iou_thresh': 0.0, 'crop_to_roi': True, 'use_shapely': True,
                'verbose': True},
            verbose=True, monitorPrefix="test",
        )

        left = 59206
        top = 33505
        expected_savename = (
            'TCGA-A2-A0YE-01Z-00-DX1.8A2E3094-5755-42BC-969D-7F0A2ECA0F39_'
            'left-%d_top-%d_mag-BASE.png' % (
                left, top))
        ROI = imread(os.path.join(mask_savepath, expected_savename))

        assert ROI.shape == (4594, 4542)
