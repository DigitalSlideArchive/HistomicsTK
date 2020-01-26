import unittest

# import os
# import matplotlib.pylab as plt
import girder_client
from histomicstk.annotations_and_masks.annotation_and_mask_utils import \
    get_scale_factor_and_appendStr, scale_slide_annotations, \
    get_bboxes_from_slide_annotations
from histomicstk.annotations_and_masks.annotations_to_object_mask_handler import \
    annotations_to_contours_no_mask

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

# get annotations for slide
slide_annotations = gc.get('/annotation/item/' + SAMPLE_SLIDE_ID)

# scale up/down annotations by a factor
sf, _ = get_scale_factor_and_appendStr(
    gc=gc, slide_id=SAMPLE_SLIDE_ID, MPP=MPP, MAG=MAG)
slide_annotations = scale_slide_annotations(slide_annotations, sf=sf)

# get bounding box information for all annotations
element_infos = get_bboxes_from_slide_annotations(slide_annotations)

# %%===========================================================================

# params for annotations_to_contours_no_mask()
get_kwargs = {
    'gc': gc,
    'slide_id': SAMPLE_SLIDE_ID,
    'MPP': MPP, 'MAG': MAG,
    'bounds': {
        'XMIN': 58000, 'XMAX': 63000,
        'YMIN': 35000, 'YMAX': 39000},
    'linewidth': 0.2,
    'get_rgb': True, 'get_visualization': True,
}

# %%===========================================================================


class GetSlideRegionNoMask(unittest.TestCase):
    """Test methods for getting ROI contours from annotations."""

    def test_annotations_to_contours_no_mask_1(self):
        """Test annotations_to_contours_no_mask()."""
        print("test_annotations_to_contours_no_mask_1()")

        # get specified region -- without providing scaled annotations
        roi_out_1 = annotations_to_contours_no_mask(
            mode='manual_bounds', **get_kwargs)

        # get specified region -- with providing scaled annotations
        roi_out_2 = annotations_to_contours_no_mask(
            mode='manual_bounds', slide_annotations=slide_annotations,
            element_infos=element_infos, **get_kwargs)

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
                {'annidx', 'elementidx', 'element_girder_id', 'type',
                 'annotation_girder_id', 'bbox_area', 'group', 'color',
                 'ymin', 'ymax', 'xmin', 'xmax', 'coords_x', 'coords_y'})

    def test_annotations_to_contours_no_mask_2(self):
        """Test get_image_and_mask_from_slide()."""
        print("test_get_image_and_mask_from_slide_2()")

        # get ROI bounding everything
        minbbox_out = annotations_to_contours_no_mask(
            mode='min_bounding_box', slide_annotations=slide_annotations,
            element_infos=element_infos, **get_kwargs)

        self.assertSetEqual(
            set(minbbox_out.keys()),
            {'bounds', 'rgb', 'contours', 'visualization'})
        self.assertTupleEqual(minbbox_out['rgb'].shape, (321, 351, 3))
        self.assertTupleEqual(
            minbbox_out['visualization'].shape, (321, 351, 3))
        self.assertAlmostEqual(len(minbbox_out['contours']) * 0.01, 0.76, 1)
        self.assertSetEqual(
            set(minbbox_out['contours'][0].keys()),
            {'annidx', 'elementidx', 'element_girder_id', 'type',
             'annotation_girder_id', 'bbox_area', 'group', 'color',
             'ymin', 'ymax', 'xmin', 'xmax', 'coords_x', 'coords_y'})

    def test_annotations_to_contours_no_mask_3(self):
        """Test get_image_and_mask_from_slide()."""
        print("test_get_image_and_mask_from_slide_3()")

        # get entire wsi region
        wsi_out = annotations_to_contours_no_mask(
            mode='wsi', slide_annotations=slide_annotations,
            element_infos=element_infos, **get_kwargs)

        self.assertSetEqual(
            set(wsi_out.keys()),
            {'bounds', 'rgb', 'contours', 'visualization'})
        self.assertTupleEqual(wsi_out['rgb'].shape, (4030, 6589, 3))
        self.assertTupleEqual(
            wsi_out['visualization'].shape, (4030, 6589, 3))
        self.assertAlmostEqual(len(wsi_out['contours']) * 0.01, 0.76, 1)
        self.assertSetEqual(
            set(wsi_out['contours'][0].keys()),
            {'annidx', 'elementidx', 'element_girder_id', 'type',
             'annotation_girder_id', 'bbox_area', 'group', 'color',
             'ymin', 'ymax', 'xmin', 'xmax', 'coords_x', 'coords_y'})


# %%===========================================================================


if __name__ == '__main__':

    unittest.main()
