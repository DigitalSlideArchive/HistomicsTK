# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 18:47:34 2019.

@author: tageldim

"""
import unittest

import os
import girder_client
from pandas import read_csv
import tempfile
import shutil

from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    get_bboxes_from_slide_annotations, _get_idxs_for_all_rois,
    scale_slide_annotations, get_scale_factor_and_appendStr)
from histomicstk.annotations_and_masks.annotations_to_masks_handler import (
    get_roi_mask, get_image_and_mask_from_slide, get_all_rois_from_slide)

# %%===========================================================================
# Constants & prep work
# =============================================================================

APIURL = 'http://candygram.neurology.emory.edu:8080/api/v1/'
SAMPLE_SLIDE_ID = '5d586d57bd4404c6b1f28640'
GTCODE_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_files', 'sample_GTcodes.csv')

gc = girder_client.GirderClient(apiUrl=APIURL)
# gc.authenticate(interactive=True)
gc.authenticate(apiKey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')

# read ground truth codes and information
GTCodes = read_csv(GTCODE_PATH)
GTCodes.index = GTCodes.loc[:, 'group']

# just a temp directory to save masks for now
BASE_SAVEPATH = tempfile.mkdtemp()
SAVEPATHS = {
    'ROI': os.path.join(BASE_SAVEPATH, 'masks'),
    'rgb': os.path.join(BASE_SAVEPATH, 'rgbs'),
    'contours': os.path.join(BASE_SAVEPATH, 'contours'),
    'visualization': os.path.join(BASE_SAVEPATH, 'vis'),
}
for _, savepath in SAVEPATHS.items():
    os.mkdir(savepath)

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
slide_annotations = gc.get('/annotation/item/' + SAMPLE_SLIDE_ID)

# scale up/down annotations by a factor
sf, _ = get_scale_factor_and_appendStr(
    gc=gc, slide_id=SAMPLE_SLIDE_ID, MPP=MPP, MAG=MAG)
slide_annotations = scale_slide_annotations(slide_annotations, sf=sf)

# get bounding box information for all annotations
element_infos = get_bboxes_from_slide_annotations(slide_annotations)

# params for get_image_and_mask_from_slide()
get_kwargs = {
    'gc': gc, 'slide_id': SAMPLE_SLIDE_ID,
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

# %%===========================================================================


class GetROIMasksTest(unittest.TestCase):
    """Test methods for getting ROI mask from annotations."""

    def test_get_roi_mask(self):
        """Test get_roi_mask()."""
        print("test_get_roi_mask()")

        # get indices of rois
        idxs_for_all_rois = _get_idxs_for_all_rois(
            GTCodes=GTCodes, element_infos=element_infos)

        # get roi mask and info
        ROI, roiinfo = get_roi_mask(
            slide_annotations=slide_annotations, element_infos=element_infos,
            GTCodes_df=GTCodes.copy(),
            idx_for_roi=idxs_for_all_rois[0],  # <- let's focus on first ROI,
            roiinfo=None, **get_roi_mask_kwargs)

        self.assertTupleEqual(ROI.shape, (228, 226))
        self.assertTupleEqual((
            roiinfo['BBOX_HEIGHT'], roiinfo['BBOX_WIDTH'],
            roiinfo['XMIN'], roiinfo['XMAX'],
            roiinfo['YMIN'], roiinfo['YMAX']),
            (242, 351, 2966, 3317, 1678, 1920))

    def test_get_all_rois_from_slide(self):
        """Test get_all_rois_from_slide()."""
        print("test_get_all_rois_from_slide()")

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
            gc=gc, slide_id=SAMPLE_SLIDE_ID, GTCodes_dict=GTCodes.T.to_dict(),
            save_directories=SAVEPATHS,
            get_image_and_mask_from_slide_kwargs=detailed_kwargs,
            slide_name='TCGA-A2-A0YE', verbose=False)

        self.assertEqual(len(savenames), 3)
        self.assertSetEqual(
            set(savenames[0].keys()),
            {'ROI', 'rgb', 'visualization', 'contours'})
        self.assertEqual(
            os.path.basename(savenames[0]['ROI']),
            "TCGA-A2-A0YE_left-59201_top-33493_bottom-63732_right-38063.png")

# %%===========================================================================


class GetSlideRegion(unittest.TestCase):
    """Test methods for getting ROI mask from annotations."""

    def test_get_image_and_mask_from_slide_1(self):
        """Test get_image_and_mask_from_slide()."""
        print("test_get_image_and_mask_from_slide_1()")

        # get specified region -- without providing scaled annotations
        roi_out_1 = get_image_and_mask_from_slide(
            mode='manual_bounds', **get_kwargs)

        # get specified region -- with providing scaled annotations
        roi_out_2 = get_image_and_mask_from_slide(
            mode='manual_bounds', slide_annotations=slide_annotations,
            element_infos=element_infos, **get_kwargs)

        for roi_out in (roi_out_1, roi_out_2):
            self.assertSetEqual(
                set(roi_out.keys()),
                {'bounds', 'ROI', 'rgb', 'contours', 'visualization'})
            self.assertTupleEqual(roi_out['ROI'].shape, (200, 250))
            self.assertTupleEqual(roi_out['rgb'].shape, (200, 250, 3))
            self.assertTupleEqual(
                roi_out['visualization'].shape, (200, 250, 3))
            self.assertEqual(len(roi_out['contours']), 29)
            self.assertSetEqual(
                set(roi_out['contours'][0].keys()),
                {'group', 'color', 'ymin', 'ymax', 'xmin', 'xmax',
                 'has_holes', 'touches_edge-top', 'touches_edge-left',
                 'touches_edge-bottom', 'touches_edge-right', 'coords_x',
                 'coords_y'})

    def test_get_image_and_mask_from_slide_2(self):
        """Test get_image_and_mask_from_slide()."""
        print("test_get_image_and_mask_from_slide_2()")

        # get ROI bounding everything
        minbbox_out = get_image_and_mask_from_slide(
            mode='min_bounding_box', slide_annotations=slide_annotations,
            element_infos=element_infos, **get_kwargs)

        self.assertSetEqual(
            set(minbbox_out.keys()),
            {'bounds', 'ROI', 'rgb', 'contours', 'visualization'})
        self.assertTupleEqual(minbbox_out['ROI'].shape, (321, 351))
        self.assertTupleEqual(minbbox_out['rgb'].shape, (321, 351, 3))
        self.assertTupleEqual(
            minbbox_out['visualization'].shape, (321, 351, 3))
        self.assertEqual(len(minbbox_out['contours']), 29)
        self.assertSetEqual(
            set(minbbox_out['contours'][0].keys()),
            {'group', 'color', 'ymin', 'ymax', 'xmin', 'xmax',
             'has_holes', 'touches_edge-top', 'touches_edge-left',
             'touches_edge-bottom', 'touches_edge-right', 'coords_x',
             'coords_y'})

    def test_get_image_and_mask_from_slide_3(self):
        """Test get_image_and_mask_from_slide()."""
        print("test_get_image_and_mask_from_slide_3()")

        # get entire wsi region
        wsi_out = get_image_and_mask_from_slide(
            mode='wsi', slide_annotations=slide_annotations,
            element_infos=element_infos, **get_kwargs)

        self.assertSetEqual(
            set(wsi_out.keys()),
            {'bounds', 'ROI', 'rgb', 'contours', 'visualization'})
        self.assertTupleEqual(wsi_out['ROI'].shape, (4030, 6590))
        self.assertTupleEqual(wsi_out['rgb'].shape, (4030, 6590, 3))
        self.assertTupleEqual(
            wsi_out['visualization'].shape, (4030, 6590, 3))
        self.assertEqual(len(wsi_out['contours']), 30)
        self.assertSetEqual(
            set(wsi_out['contours'][0].keys()),
            {'group', 'color', 'ymin', 'ymax', 'xmin', 'xmax',
             'has_holes', 'touches_edge-top', 'touches_edge-left',
             'touches_edge-bottom', 'touches_edge-right', 'coords_x',
             'coords_y'})

# %%===========================================================================


if __name__ == '__main__':

    unittest.main()

    # cleanup
    shutil.rmtree(BASE_SAVEPATH)
