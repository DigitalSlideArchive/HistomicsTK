import copy
import unittest

import os
import tempfile
import shutil
# import matplotlib.pylab as plt
from imageio import imread
from pandas import read_csv
import numpy as np
import girder_client
from histomicstk.annotations_and_masks.annotation_and_mask_utils import \
    get_scale_factor_and_appendStr, scale_slide_annotations, \
    get_bboxes_from_slide_annotations
from histomicstk.annotations_and_masks.annotations_to_object_mask_handler \
    import annotations_to_contours_no_mask, get_all_rois_from_slide_v2

# %%===========================================================================
# Constants & prep work

APIURL = 'http://candygram.neurology.emory.edu:8080/api/v1/'
SAMPLE_SLIDE_ID = '5d586d57bd4404c6b1f28640'

gc = girder_client.GirderClient(apiUrl=APIURL)
# gc.authenticate(interactive=True)
gc.authenticate(apiKey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')

# Microns-per-pixel / Magnification (either or)
MPP = 5.0
MAG = None

# GT codes dict for parsing into label mask
GTCODE_PATH = os.path.join(
    '/home/mtageld/Desktop/HistomicsTK/histomicstk/annotations_and_masks/',
    'tests/test_files', 'sample_GTcodes.csv')
GTCodes_dict = read_csv(GTCODE_PATH)
GTCodes_dict.index = GTCodes_dict.loc[:, 'group']
GTCodes_dict = GTCodes_dict.to_dict(orient='index')

# just a temp directory to save masks for now
BASE_SAVEPATH = tempfile.mkdtemp()
SAVEPATHS = {
    'contours': os.path.join(BASE_SAVEPATH, 'contours'),
    'rgb': os.path.join(BASE_SAVEPATH, 'rgbs'),
    'visualization': os.path.join(BASE_SAVEPATH, 'vis'),
    'mask': os.path.join(BASE_SAVEPATH, 'masks'),
}
for _, savepath in SAVEPATHS.items():
    if not os.path.exists(savepath):
        os.mkdir(savepath)

# get annotations for slide
slide_annotations = gc.get('/annotation/item/' + SAMPLE_SLIDE_ID)

# scale up/down annotations by a factor
sf, _ = get_scale_factor_and_appendStr(
    gc=gc, slide_id=SAMPLE_SLIDE_ID, MPP=MPP, MAG=MAG)
slide_annotations = scale_slide_annotations(slide_annotations, sf=sf)

# get bounding box information for all annotations
element_infos = get_bboxes_from_slide_annotations(slide_annotations)

# %%===========================================================================

# common params for annotations_to_contours_no_mask()
annotations_to_contours_kwargs = {
    'MPP': MPP, 'MAG': MAG,
    'linewidth': 0.2,
    'get_rgb': True, 'get_visualization': True,
}

# params for TESTING annotations_to_contours_no_mask()
test_annots_to_contours_kwargs = copy.deepcopy(
    annotations_to_contours_kwargs)
test_annots_to_contours_kwargs.update({
    'gc': gc,
    'slide_id': SAMPLE_SLIDE_ID,
    'bounds': {
        'XMIN': 58000, 'XMAX': 63000,
        'YMIN': 35000, 'YMAX': 39000},
    })

# params for getting all rois for slide
get_all_rois_kwargs = {
    'gc': gc,
    'slide_id': SAMPLE_SLIDE_ID,
    'GTCodes_dict': GTCodes_dict,
    'save_directories': SAVEPATHS,
    'annotations_to_contours_kwargs': annotations_to_contours_kwargs,
    'slide_name': 'TCGA-A2-A0YE',
    'verbose': False,
    'monitorprefix': 'test',
}

# %%===========================================================================


class GetSlideRegionNoMask(unittest.TestCase):
    """Test methods for getting ROI contours from annotations."""

    def test_annotations_to_contours_no_mask_1(self):
        """Test annotations_to_contours_no_mask()."""
        print("test_annotations_to_contours_no_mask_1()")

        # get specified region -- without providing scaled annotations
        roi_out_1 = annotations_to_contours_no_mask(
            mode='manual_bounds', **test_annots_to_contours_kwargs)

        # get specified region -- with providing scaled annotations
        roi_out_2 = annotations_to_contours_no_mask(
            mode='manual_bounds', slide_annotations=slide_annotations,
            element_infos=element_infos, **test_annots_to_contours_kwargs)

        for roi_out in (roi_out_1, roi_out_2):
            self.assertSetEqual(
                set(roi_out.keys()),
                {'bounds', 'rgb', 'contours', 'visualization'})
            self.assertTupleEqual(roi_out['rgb'].shape, (200, 251, 3))
            self.assertTupleEqual(
                roi_out['visualization'].shape, (200, 251, 3))
            self.assertAlmostEqual(len(roi_out['contours']) * 0.01, 0.64, 1)
            self.assertSetEqual(
                set(roi_out['contours'][0].keys()),
                {'annidx', 'elementidx', 'element_girder_id', 'type', 'label',
                 'annotation_girder_id', 'bbox_area', 'group', 'color',
                 'ymin', 'ymax', 'xmin', 'xmax', 'coords_x', 'coords_y'})

    def test_annotations_to_contours_no_mask_2(self):
        """Test get_image_and_mask_from_slide()."""
        print("test_get_image_and_mask_from_slide_2()")

        # get ROI bounding everything
        minbbox_out = annotations_to_contours_no_mask(
            mode='min_bounding_box', slide_annotations=slide_annotations,
            element_infos=element_infos, **test_annots_to_contours_kwargs)

        self.assertSetEqual(
            set(minbbox_out.keys()),
            {'bounds', 'rgb', 'contours', 'visualization'})
        self.assertTupleEqual(minbbox_out['rgb'].shape, (321, 351, 3))
        self.assertTupleEqual(
            minbbox_out['visualization'].shape, (321, 351, 3))
        self.assertAlmostEqual(len(minbbox_out['contours']) * 0.01, 0.76, 1)
        self.assertSetEqual(
            set(minbbox_out['contours'][0].keys()),
            {'annidx', 'elementidx', 'element_girder_id', 'type', 'label',
             'annotation_girder_id', 'bbox_area', 'group', 'color',
             'ymin', 'ymax', 'xmin', 'xmax', 'coords_x', 'coords_y'})

    # def test_annotations_to_contours_no_mask_3(self):
    #     """Test get_image_and_mask_from_slide()."""
    #     print("test_get_image_and_mask_from_slide_3()")
    #
    #     # get entire wsi region
    #     wsi_out = annotations_to_contours_no_mask(
    #         mode='wsi', slide_annotations=slide_annotations,
    #         element_infos=element_infos,
    #         **test_annots_to_contours_kwargs)
    #
    #     self.assertSetEqual(
    #         set(wsi_out.keys()),
    #         {'bounds', 'rgb', 'contours', 'visualization'})
    #     self.assertTupleEqual(wsi_out['rgb'].shape, (4030, 6589, 3))
    #     self.assertTupleEqual(
    #         wsi_out['visualization'].shape, (4030, 6589, 3))
    #     self.assertAlmostEqual(len(wsi_out['contours']) * 0.01, 0.76, 1)
    #     self.assertSetEqual(
    #         set(wsi_out['contours'][0].keys()),
    #         {'annidx', 'elementidx', 'element_girder_id', 'type', 'label',
    #          'annotation_girder_id', 'bbox_area', 'group', 'color',
    #          'ymin', 'ymax', 'xmin', 'xmax', 'coords_x', 'coords_y'})


# %%===========================================================================


class GetSlideRoisNoMask(unittest.TestCase):
    """Test methods for getting all ROIs without intermediate masks."""

    def test_get_all_rois_from_slide_v2(self):
        """Test get_all_rois_from_slide_v2()."""
        print("test_get_all_rois_from_slide_v2()")

        # First we test the object segmentation mode
        get_all_rois_kwargs['mode'] = 'object'
        savenames = get_all_rois_from_slide_v2(**get_all_rois_kwargs)

        # basic checks
        self.assertEqual(len(savenames), 3)
        self.assertSetEqual(
            set(savenames[0].keys()),
            {'mask', 'rgb', 'visualization', 'contours'})
        self.assertSetEqual(
            {'TCGA-A2-A0YE_left-58463_top-38203_bottom-39760_right-60379.png',
             'TCGA-A2-A0YE_left-59181_top-33473_bottom-38043_right-63712.png',
             'TCGA-A2-A0YE_left-57584_top-35788_bottom-37425_right-59421.png',
             },
            {os.path.basename(savename['mask']) for savename in savenames})

        # shape & value check
        imname = 'TCGA-A2-A0YE_left-57584_top-35788_bottom-37425_right-59421'
        mask = imread(os.path.join(SAVEPATHS['mask'], imname + '.png'))
        self.assertTupleEqual(mask.shape, (82, 92, 3))
        self.assertSetEqual(set(np.unique(mask[..., 0])), {0, 1, 2, 7})
        self.assertSetEqual(set(np.unique(mask[..., 1])), {0, 1})
        self.assertSetEqual(
            set(np.unique(mask[..., 2])),
            {0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})

        # Second, we test the semantic segmentation mode
        get_all_rois_kwargs['mode'] = 'semantic'
        savenames = get_all_rois_from_slide_v2(**get_all_rois_kwargs)

        # shape check
        mask = imread(os.path.join(SAVEPATHS['mask'], imname + '.png'))
        self.assertTupleEqual(mask.shape, (82, 92))

# %%===========================================================================


if __name__ == '__main__':

    unittest.main()

    # cleanup
    shutil.rmtree(BASE_SAVEPATH)
