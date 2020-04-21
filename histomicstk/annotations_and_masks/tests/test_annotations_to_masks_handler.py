# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 18:47:34 2019.

@author: tageldim

"""
import os
import copy
import pytest
from pandas import read_csv
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    get_bboxes_from_slide_annotations, _get_idxs_for_all_rois,
    scale_slide_annotations, get_scale_factor_and_appendStr)
from histomicstk.annotations_and_masks.annotations_to_masks_handler import (
    get_roi_mask, get_image_and_mask_from_slide, get_all_rois_from_slide)

import sys
thisDir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(thisDir, '../../../tests'))
import htk_test_utilities as utilities  # noqa
from htk_test_utilities import girderClient, getTestFilePath  # noqa

# # for protyping
# from tests.htk_test_utilities import _connect_to_existing_local_dsa
# girderClient = _connect_to_existing_local_dsa()

global gc, GTcodes, iteminfo, slide_annotations, element_infos, \
    get_roi_mask_kwargs, get_contours_kwargs, get_kwargs, MPP, MAG


# pytest runs tests in the order they appear in the module
@pytest.mark.usefixtures('girderClient')  # noqa
def test_prep(girderClient):  # noqa
    global gc, GTCodes, iteminfo, slide_annotations, element_infos, \
        get_roi_mask_kwargs, get_contours_kwargs, get_kwargs, MPP, MAG

    gc = girderClient

    iteminfo = gc.get('/item', parameters={
        'text': "TCGA-A2-A0YE-01Z-00-DX1"})[0]

    # read GTCodes dataframe
    gtcodePath = getTestFilePath('sample_GTcodes.csv')
    GTCodes = read_csv(gtcodePath)
    GTCodes.index = GTCodes.loc[:, 'group']

    # other params
    get_roi_mask_kwargs = {
        'iou_thresh': 0.0,
        'crop_to_roi': True,
        'use_shapely': True,
        'verbose': False
    }
    get_contours_kwargs = {
        'groups_to_get': None,
        'roi_group': 'roi',
        'get_roi_contour': True,
        'discard_nonenclosed_background': True,
        'background_group': 'mostly_stroma',
        'MIN_SIZE': 10, 'MAX_SIZE': None,
        'verbose': False, 'monitorPrefix': ""
    }

    # Microns-per-pixel / Magnification (either or)
    MPP = 5.0
    MAG = None

    # get annotations for slide
    slide_annotations = gc.get('/annotation/item/' + iteminfo['_id'])

    # scale up/down annotations by a factor
    sf, _ = get_scale_factor_and_appendStr(
        gc=gc, slide_id=iteminfo['_id'], MPP=MPP, MAG=MAG)
    slide_annotations = scale_slide_annotations(slide_annotations, sf=sf)

    # get bounding box information for all annotations
    element_infos = get_bboxes_from_slide_annotations(slide_annotations)

    # params for get_image_and_mask_from_slide()
    get_kwargs = {
        'gc': gc, 'slide_id': iteminfo['_id'],
        'GTCodes_dict': GTCodes.T.to_dict(),
        'bounds': {
            'XMIN': 58000, 'XMAX': 63000,
            'YMIN': 35000, 'YMAX': 39000},
        'MPP': MPP,
        'MAG': MAG,
        'get_roi_mask_kwargs': get_roi_mask_kwargs,
        'get_contours_kwargs': get_contours_kwargs,
        'get_rgb': True,
        'get_contours': True,
        'get_visualization': True,
    }


class TestGetROIMasks(object):
    """Test methods for getting ROI mask from annotations."""

    def test_get_roi_mask(self):
        """Test get_roi_mask()."""
        # get indices of rois
        idxs_for_all_rois = _get_idxs_for_all_rois(
            GTCodes=GTCodes, element_infos=element_infos.copy())

        # get roi mask and info
        ROI, roiinfo = get_roi_mask(
            slide_annotations=copy.deepcopy(slide_annotations),
            element_infos=element_infos.copy(),
            GTCodes_df=GTCodes.copy(),
            idx_for_roi=idxs_for_all_rois[0],  # <- let's focus on first ROI,
            roiinfo=None, **get_roi_mask_kwargs)

        assert ROI.shape == (228, 226)
        assert (
            roiinfo['BBOX_HEIGHT'], roiinfo['BBOX_WIDTH'],
            roiinfo['XMIN'], roiinfo['XMAX'],
            roiinfo['YMIN'], roiinfo['YMAX']) == (
            242, 351, 2966, 3317, 1678, 1920)

    def test_get_all_rois_from_slide(self, tmpdir):  # noqa
        """Test get_all_roi_masks_for_slide()."""
        # just a temp directory to save masks for now
        base_savepath = str(tmpdir)
        savepaths = {
            'ROI': os.path.join(base_savepath, 'masks'),
            'rgb': os.path.join(base_savepath, 'rgbs'),
            'contours': os.path.join(base_savepath, 'contours'),
            'visualization': os.path.join(base_savepath, 'vis'),
        }
        for _, savepath in savepaths.items():
            os.mkdir(savepath)

        detailed_kwargs = {
            'MPP': MPP,
            'MAG': None,
            'get_roi_mask_kwargs': get_roi_mask_kwargs,
            'get_contours_kwargs': get_contours_kwargs,
            'get_rgb': True,
            'get_contours': True,
            'get_visualization': True,
        }

        savenames = get_all_rois_from_slide(
            gc=gc, slide_id=iteminfo['_id'],
            GTCodes_dict=GTCodes.T.to_dict(), save_directories=savepaths,
            get_image_and_mask_from_slide_kwargs=detailed_kwargs,
            slide_name='TCGA-A2-A0YE', verbose=False)

        assert len(savenames) == 3
        assert set(savenames[0].keys()) == {
            'ROI', 'rgb', 'visualization', 'contours'}
        assert {
            'TCGA-A2-A0YE_left-57604_top-35808_bottom-37445_right-59441.png',
            'TCGA-A2-A0YE_left-58483_top-38223_bottom-39780_right-60399.png',
            'TCGA-A2-A0YE_left-59201_top-33493_bottom-38063_right-63732.png'
        } == {os.path.basename(savename['ROI']) for savename in savenames}

    def test_get_image_and_mask_manual_bounds(self):
        """Test get_image_and_mask_from_slide()."""
        # get specified region -- without providing scaled annotations
        roi_out_1 = get_image_and_mask_from_slide(
            mode='manual_bounds', **get_kwargs)

        # get specified region -- with providing scaled annotations
        roi_out_2 = get_image_and_mask_from_slide(
            mode='manual_bounds',
            slide_annotations=copy.deepcopy(slide_annotations),
            element_infos=element_infos.copy(), **get_kwargs)

        for roi_out in (roi_out_1, roi_out_2):
            assert set(roi_out.keys()) == {
                'bounds', 'ROI', 'rgb', 'contours', 'visualization'}
            assert roi_out['ROI'].shape == (200, 250)
            assert roi_out['rgb'].shape == (200, 250, 3)
            assert roi_out['visualization'].shape == (200, 250, 3)
            assert len(roi_out['contours']) > 26 and (
                len(roi_out['contours']) < 32)
            assert set(roi_out['contours'][0].keys()) == {
                'group', 'color', 'ymin', 'ymax', 'xmin', 'xmax',
                'has_holes', 'touches_edge-top', 'touches_edge-left',
                'touches_edge-bottom', 'touches_edge-right', 'coords_x',
                'coords_y'
            }

    def test_get_image_and_mask_minbbox(self):
        """Test get_image_and_mask_from_slide()."""
        # get ROI bounding everything
        roi_out = get_image_and_mask_from_slide(
            mode='min_bounding_box',
            slide_annotations=copy.deepcopy(slide_annotations),
            element_infos=element_infos.copy(), **get_kwargs)

        assert set(roi_out.keys()) == {
            'bounds', 'ROI', 'rgb', 'contours', 'visualization'}
        assert roi_out['ROI'].shape == (321, 351)
        assert roi_out['rgb'].shape == (321, 351, 3)
        assert roi_out['visualization'].shape == (321, 351, 3)
        assert len(roi_out['contours']) > 26 and (
                len(roi_out['contours']) < 32)
        assert set(roi_out['contours'][0].keys()) == {
            'group', 'color', 'ymin', 'ymax', 'xmin', 'xmax',
            'has_holes', 'touches_edge-top', 'touches_edge-left',
            'touches_edge-bottom', 'touches_edge-right', 'coords_x',
            'coords_y'
        }
