import unittest

# import os
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
MPP = 2.5  # 5.0
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
    'linewidth': 0.2, 'crop_to_roi': True,
    'get_rgb': True, 'get_visualization': True,
}

# %%===========================================================================

roi_out = annotations_to_contours_no_mask(
    mode='manual_bounds', **get_kwargs)


# %%===========================================================================
#
# class GetSlideRegionNoMask(unittest.TestCase):
#     """Test methods for getting ROI contours from annotations."""
#
#     def test_annotations_to_contours_no_mask_1(self):
#         """Test annotations_to_contours_no_mask()."""
#         print("test_annotations_to_contours_no_mask_1()")
#
#         # get specified region -- without providing scaled annotations
#         roi_out_1 = annotations_to_contours_no_mask(
#             mode='manual_bounds', **get_kwargs)
#
#         # get specified region -- with providing scaled annotations
#         roi_out_2 = annotations_to_contours_no_mask(
#             mode='manual_bounds', slide_annotations=slide_annotations,
#             element_infos=element_infos, **get_kwargs)
#
#         for roi_out in (roi_out_1, roi_out_2):
#             self.assertSetEqual(
#                 set(roi_out.keys()),
#                 {'bounds', 'rgb', 'contours', 'visualization'})
#             self.assertTupleEqual(roi_out['rgb'].shape, (321, 351, 3))
#             self.assertTupleEqual(
#                 roi_out['visualization'].shape, (200, 250, 3))
#             self.assertAlmostEqual(len(roi_out['contours']) * 0.01, 0.25, 1)
#             self.assertSetEqual(
#                 set(roi_out['contours'][0].keys()),
#                 {'group', 'color', 'ymin', 'ymax', 'xmin', 'xmax',
#                  'has_holes', 'touches_edge-top', 'touches_edge-left',
#                  'touches_edge-bottom', 'touches_edge-right', 'coords_x',
#                  'coords_y'})
#
#
#
# # %%===========================================================================
#
#
#
#
#
#
