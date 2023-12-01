import copy
import os
import shutil
import sys
import tempfile

import numpy as np
import pytest
from imageio import imread
from pandas import read_csv

from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    get_bboxes_from_slide_annotations, get_scale_factor_and_appendStr,
    scale_slide_annotations)
from histomicstk.annotations_and_masks.annotations_to_object_mask_handler import (
    annotations_to_contours_no_mask, get_all_rois_from_slide_v2)

thisDir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(thisDir, '../../../tests'))
import htk_test_utilities as utilities  # noqa
from htk_test_utilities import getTestFilePath, girderClient  # noqa


class Cfg:
    def __init__(self):
        self.gc = None
        self.slide_annotations = None
        self.element_infos = None
        self.BASE_SAVEPATH = None
        self.SAVEPATHS = None
        self.test_annots_to_contours_kwargs = None
        self.get_all_rois_kwargs = None
        self.MPP = None
        self.MAG = None


cfg = Cfg()

# pytest runs tests in the order they appear in the module


@pytest.mark.usefixtures('girderClient')  # noqa
def test_prep(girderClient):  # noqa

    cfg.gc = girderClient

    iteminfo = cfg.gc.get('/item', parameters={'text': 'TCGA-A2-A0YE-01Z-00-DX1'})[0]

    # read GTCodes dataframe
    gtcodePath = getTestFilePath('sample_GTcodes.csv')
    GTCodes_dict = read_csv(gtcodePath)
    GTCodes_dict.index = GTCodes_dict.loc[:, 'group']
    GTCodes_dict = GTCodes_dict.to_dict(orient='index')

    # just a temp directory to save masks for now
    cfg.BASE_SAVEPATH = tempfile.mkdtemp()
    cfg.SAVEPATHS = {
        'contours': os.path.join(cfg.BASE_SAVEPATH, 'contours'),
        'rgb': os.path.join(cfg.BASE_SAVEPATH, 'rgbs'),
        'visualization': os.path.join(cfg.BASE_SAVEPATH, 'vis'),
        'mask': os.path.join(cfg.BASE_SAVEPATH, 'masks'),
    }
    for _, savepath in cfg.SAVEPATHS.items():
        if not os.path.exists(savepath):
            os.mkdir(savepath)

    # Microns-per-pixel / Magnification (either or)
    cfg.MPP = 5.0
    cfg.MAG = None

    # get annotations for slide
    cfg.slide_annotations = cfg.gc.get('/annotation/item/' + iteminfo['_id'])

    # scale up/down annotations by a factor
    sf, _ = get_scale_factor_and_appendStr(
        gc=cfg.gc, slide_id=iteminfo['_id'], MPP=cfg.MPP, MAG=cfg.MAG)
    cfg.slide_annotations = scale_slide_annotations(cfg.slide_annotations, sf=sf)

    # get bounding box information for all annotations
    cfg.element_infos = get_bboxes_from_slide_annotations(cfg.slide_annotations)

    # common params for annotations_to_contours_no_mask()
    annotations_to_contours_kwargs = {
        'MPP': cfg.MPP, 'MAG': cfg.MAG,
        'linewidth': 0.2,
        'get_rgb': True, 'get_visualization': True,
    }

    # params for TESTING annotations_to_contours_no_mask()
    cfg.test_annots_to_contours_kwargs = copy.deepcopy(
        annotations_to_contours_kwargs)
    cfg.test_annots_to_contours_kwargs.update({
        'gc': cfg.gc,
        'slide_id': iteminfo['_id'],
        'bounds': {
            'XMIN': 58000, 'XMAX': 63000,
            'YMIN': 35000, 'YMAX': 39000},
    })

    # params for getting all rois for slide
    cfg.get_all_rois_kwargs = {
        'gc': cfg.gc,
        'slide_id': iteminfo['_id'],
        'GTCodes_dict': GTCodes_dict,
        'save_directories': cfg.SAVEPATHS,
        'annotations_to_contours_kwargs': annotations_to_contours_kwargs,
        'slide_name': 'TCGA-A2-A0YE',
        'verbose': False,
        'monitorprefix': 'test',
    }


class TestGetSlideRegionNoMask:
    """Test methods for getting ROI contours from annotations."""

    def test_annotations_to_contours_no_mask_1(self):
        """Test annotations_to_contours_no_mask()."""
        # get specified region -- without providing scaled annotations
        roi_out_1 = annotations_to_contours_no_mask(
            mode='manual_bounds', **cfg.test_annots_to_contours_kwargs)

        # get specified region -- with providing scaled annotations
        roi_out_2 = annotations_to_contours_no_mask(
            mode='manual_bounds', slide_annotations=cfg.slide_annotations,
            element_infos=cfg.element_infos, **cfg.test_annots_to_contours_kwargs)

        for roi_out in (roi_out_1, roi_out_2):
            assert set(roi_out.keys()) == {
                'bounds', 'rgb', 'contours', 'visualization'}
            assert roi_out['rgb'].shape == (200, 251, 3)
            assert roi_out['visualization'].shape == (200, 251, 3)
            assert len(roi_out['contours']) > 56
            assert (
                len(roi_out['contours']) < 68)
            assert set(roi_out['contours'][0].keys()) == {
                'annidx', 'elementidx', 'element_girder_id', 'type', 'label',
                'annotation_girder_id', 'bbox_area', 'group', 'color',
                'ymin', 'ymax', 'xmin', 'xmax', 'coords_x', 'coords_y',
            }

    def test_annotations_to_contours_no_mask_2(self):
        """Test get_image_and_mask_from_slide()."""
        # get ROI bounding everything
        roi_out = annotations_to_contours_no_mask(
            mode='min_bounding_box', slide_annotations=cfg.slide_annotations,
            element_infos=cfg.element_infos, **cfg.test_annots_to_contours_kwargs)

        assert set(roi_out.keys()) == {
            'bounds', 'rgb', 'contours', 'visualization'}
        assert roi_out['rgb'].shape == (321, 351, 3)
        assert roi_out['visualization'].shape == (321, 351, 3)
        assert len(roi_out['contours']) > 72
        assert (
            len(roi_out['contours']) < 80)
        assert set(roi_out['contours'][0].keys()) == {
            'annidx', 'elementidx', 'element_girder_id', 'type', 'label',
            'annotation_girder_id', 'bbox_area', 'group', 'color',
            'ymin', 'ymax', 'xmin', 'xmax', 'coords_x', 'coords_y',
        }

    def test_get_all_rois_from_slide_v2(self):
        """Test get_all_rois_from_slide_v2()."""
        # First we test the object segmentation mode
        cfg.get_all_rois_kwargs['mode'] = 'object'
        savenames = get_all_rois_from_slide_v2(**cfg.get_all_rois_kwargs)

        # basic checks
        assert len(savenames) == 3
        assert set(savenames[0].keys()) == {
            'mask', 'rgb', 'visualization', 'contours'}
        assert {
            'TCGA-A2-A0YE_left-58463_top-38203_bottom-39760_right-60379.png',
            'TCGA-A2-A0YE_left-59181_top-33473_bottom-38043_right-63712.png',
            'TCGA-A2-A0YE_left-57584_top-35788_bottom-37425_right-59421.png',
        }, {os.path.basename(savename['mask']) for savename in savenames}

        # shape & value check
        imname = 'TCGA-A2-A0YE_left-57584_top-35788_bottom-37425_right-59421'
        mask = imread(os.path.join(cfg.SAVEPATHS['mask'], imname + '.png'))
        assert mask.shape == (82, 92, 3)
        assert set(np.unique(mask[..., 0])) == {0, 1, 2, 7}
        assert set(np.unique(mask[..., 1])) == {0, 1}
        assert set(np.unique(mask[..., 2])) == {
            0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} or set(np.unique(mask[..., 2])) == {
            0, 2, 3, 4, 5, 6, 7, 8}

        # Second, we test the semantic segmentation mode
        cfg.get_all_rois_kwargs['mode'] = 'semantic'
        savenames = get_all_rois_from_slide_v2(**cfg.get_all_rois_kwargs)
        assert len(savenames) == 3

        # shape check
        mask = imread(os.path.join(cfg.SAVEPATHS['mask'], imname + '.png'))
        assert mask.shape == (82, 92)

        # cleanup
        shutil.rmtree(cfg.BASE_SAVEPATH)
