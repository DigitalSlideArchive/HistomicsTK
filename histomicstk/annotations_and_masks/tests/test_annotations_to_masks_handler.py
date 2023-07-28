"""
Created on Mon Aug 12 18:47:34 2019.

@author: tageldim

"""
import copy
import os
import sys

import pytest
from pandas import read_csv

from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    _get_idxs_for_all_rois, get_bboxes_from_slide_annotations,
    get_scale_factor_and_appendStr, scale_slide_annotations)
from histomicstk.annotations_and_masks.annotations_to_masks_handler import (
    get_all_rois_from_slide, get_image_and_mask_from_slide, get_roi_mask)

thisDir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(thisDir, '../../../tests'))
import htk_test_utilities as utilities  # noqa
from htk_test_utilities import getTestFilePath, girderClient  # noqa


class Cfg:
    def __init__(self):
        self.gc = None
        self.GTcodes = None
        self.iteminfo = None
        self.slide_annotations = None
        self.element_infos = None
        self.get_roi_mask_kwargs = None
        self.get_contours_kwargs = None
        self.get_kwargs = None
        self.MPP = None
        self.MAG = None


cfg = Cfg()

# pytest runs tests in the order they appear in the module


@pytest.mark.usefixtures('girderClient')  # noqa
def test_prep(girderClient):  # noqa

    cfg.gc = girderClient

    original_iteminfo = cfg.gc.get('/item', parameters={'text': 'TCGA-A2-A0YE-01Z-00-DX1'})[0]

    cfg.folder = cfg.gc.post(
        '/folder', data={
            'parentId': original_iteminfo['folderId'],
            'name': 'test-annot-and-mask-handler'
        })

    # copy the item
    cfg.iteminfo = cfg.gc.post(
        '/item/%s/copy' % original_iteminfo['_id'], data={
            'name': 'TCGA-A2-A0YE-01Z.svs',
            'copyAnnotations': True,
            'folderId': cfg.folder['_id'],
        })

    # read GTCodes dataframe
    gtcodePath = getTestFilePath('sample_GTcodes.csv')
    cfg.GTcodes = read_csv(gtcodePath)
    cfg.GTcodes.index = cfg.GTcodes.loc[:, 'group']

    # other params
    cfg.get_roi_mask_kwargs = {
        'iou_thresh': 0.0,
        'crop_to_roi': True,
        'use_shapely': True,
        'verbose': False
    }
    cfg.get_contours_kwargs = {
        'groups_to_get': None,
        'roi_group': 'roi',
        'get_roi_contour': True,
        'discard_nonenclosed_background': True,
        'background_group': 'mostly_stroma',
        'MIN_SIZE': 10, 'MAX_SIZE': None,
        'verbose': False, 'monitorPrefix': ''
    }

    # Microns-per-pixel / Magnification (either or)
    cfg.MPP = 5.0
    cfg.MAG = None

    # get annotations for slide
    cfg.slide_annotations = cfg.gc.get('/annotation/item/' + cfg.iteminfo['_id'])

    # scale up/down annotations by a factor
    sf, _ = get_scale_factor_and_appendStr(
        gc=cfg.gc, slide_id=cfg.iteminfo['_id'], MPP=cfg.MPP, MAG=cfg.MAG)
    cfg.slide_annotations = scale_slide_annotations(cfg.slide_annotations, sf=sf)

    # get bounding box information for all annotations
    cfg.element_infos = get_bboxes_from_slide_annotations(cfg.slide_annotations)

    # params for get_image_and_mask_from_slide()
    cfg.get_kwargs = {
        'gc': cfg.gc, 'slide_id': cfg.iteminfo['_id'],
        'GTCodes_dict': cfg.GTcodes.T.to_dict(),
        'bounds': {
            'XMIN': 58000, 'XMAX': 63000,
            'YMIN': 35000, 'YMAX': 39000},
        'MPP': cfg.MPP,
        'MAG': cfg.MAG,
        'get_roi_mask_kwargs': cfg.get_roi_mask_kwargs,
        'get_contours_kwargs': cfg.get_contours_kwargs,
        'get_rgb': True,
        'get_contours': True,
        'get_visualization': True,
    }


class TestGetROIMasks:
    """Test methods for getting ROI mask from annotations."""

    def test_get_roi_mask(self):
        """Test get_roi_mask()."""
        # get indices of rois
        idxs_for_all_rois = _get_idxs_for_all_rois(
            GTCodes=cfg.GTcodes, element_infos=cfg.element_infos.copy())

        # get roi mask and info
        ROI, roiinfo = get_roi_mask(
            slide_annotations=copy.deepcopy(cfg.slide_annotations),
            element_infos=cfg.element_infos.copy(),
            GTCodes_df=cfg.GTcodes.copy(),
            idx_for_roi=idxs_for_all_rois[0],  # <- let's focus on first ROI,
            roiinfo=None, **cfg.get_roi_mask_kwargs)

        assert ROI.shape == (228, 226)
        assert (
            roiinfo['BBOX_HEIGHT'], roiinfo['BBOX_WIDTH'],
            roiinfo['XMIN'], roiinfo['XMAX'],
            roiinfo['YMIN'], roiinfo['YMAX']) == (
            242, 351, 2966, 3317, 1678, 1920)

    def test_get_all_rois_from_slide(self, tmpdir):  # noqa
        """Test get_all_roi_masks_for_slide() as-is without tiling."""
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
            'MPP': cfg.MPP,
            'MAG': None,
            'get_roi_mask_kwargs': cfg.get_roi_mask_kwargs,
            'get_contours_kwargs': cfg.get_contours_kwargs,
            'get_rgb': True,
            'get_contours': True,
            'get_visualization': True,
        }

        savenames = get_all_rois_from_slide(
            gc=cfg.gc, slide_id=cfg.iteminfo['_id'],
            GTCodes_dict=cfg.GTcodes.T.to_dict(), save_directories=savepaths,
            get_image_and_mask_from_slide_kwargs=detailed_kwargs,
            max_roiside=None,
            slide_name='TCGA-A2-A0YE', verbose=False)

        assert len(savenames) == 3
        assert set(savenames[0].keys()) == {
            'ROI', 'rgb', 'visualization', 'contours'}
        assert {
            'TCGA-A2-A0YE_left-57604_top-35808_bottom-37445_right-59441.png',
            'TCGA-A2-A0YE_left-58483_top-38223_bottom-39780_right-60399.png',
            'TCGA-A2-A0YE_left-59201_top-33493_bottom-38063_right-63732.png'
        } == {os.path.basename(savename['ROI']) for savename in savenames}

    def test_get_all_rois_from_slide_tiled(self, tmpdir):  # noqa
        """Test get_all_roi_masks_for_slide() with tiling."""
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
            'MPP': cfg.MPP,
            'MAG': None,
            'get_roi_mask_kwargs': cfg.get_roi_mask_kwargs,
            'get_contours_kwargs': None,
            'get_rgb': False,
            'get_contours': False,
            'get_visualization': False,
        }

        savenames = get_all_rois_from_slide(
            gc=cfg.gc, slide_id=cfg.iteminfo['_id'],
            GTCodes_dict=cfg.GTcodes.T.to_dict(), save_directories=savepaths,
            get_image_and_mask_from_slide_kwargs=detailed_kwargs,
            max_roiside=128,
            slide_name='TCGA-A2-A0YE', verbose=False)

        assert len(savenames) == 6
        assert set(savenames[0].keys()) == {'ROI'}
        assert {
            'TCGA-A2-A0YE_left-59201_top-33493_bottom-36047_right-61756.png',
            'TCGA-A2-A0YE_left-59201_top-36047_bottom-38063_right-61756.png',
            'TCGA-A2-A0YE_left-61756_top-33493_bottom-36047_right-63732.png',
            'TCGA-A2-A0YE_left-61756_top-36047_bottom-38063_right-63732.png',
            'TCGA-A2-A0YE_left-58483_top-38223_bottom-39780_right-60399.png',
            'TCGA-A2-A0YE_left-57604_top-35808_bottom-37445_right-59441.png',
        } == {os.path.basename(savename['ROI']) for savename in savenames}

    def test_get_image_and_mask_manual_bounds(self):
        """Test get_image_and_mask_from_slide()."""
        # get specified region -- without providing scaled annotations
        roi_out_1 = get_image_and_mask_from_slide(
            mode='manual_bounds', **cfg.get_kwargs)

        # get specified region -- with providing scaled annotations
        roi_out_2 = get_image_and_mask_from_slide(
            mode='manual_bounds',
            slide_annotations=copy.deepcopy(cfg.slide_annotations),
            element_infos=cfg.element_infos.copy(), **cfg.get_kwargs)

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
            slide_annotations=copy.deepcopy(cfg.slide_annotations),
            element_infos=cfg.element_infos.copy(), **cfg.get_kwargs)

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
