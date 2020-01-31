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

# gc = girder_client.GirderClient(apiUrl=APIURL)
# # gc.authenticate(interactive=True)
# gc.authenticate(apiKey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')

# # Microns-per-pixel / Magnification (either or)
# MPP = 5.0
# MAG = None

# # get annotations for slide
# slide_annotations = gc.get('/annotation/item/' + SAMPLE_SLIDE_ID)
#
# # scale up/down annotations by a factor
# sf, _ = get_scale_factor_and_appendStr(
#     gc=gc, slide_id=SAMPLE_SLIDE_ID, MPP=MPP, MAG=MAG)
# slide_annotations = scale_slide_annotations(slide_annotations, sf=sf)
#
# # get bounding box information for all annotations
# element_infos = get_bboxes_from_slide_annotations(slide_annotations)

# %%===========================================================================

# # params for annotations_to_contours_no_mask()
# get_kwargs = {
#     'gc': gc,
#     'slide_id': SAMPLE_SLIDE_ID,
#     'MPP': MPP, 'MAG': MAG,
#     'bounds': {
#         'XMIN': 58000, 'XMAX': 63000,
#         'YMIN': 35000, 'YMAX': 39000},
#     'linewidth': 0.2,
#     'get_rgb': True, 'get_visualization': True,
# }

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


# if __name__ == '__main__':
#
#     unittest.main()

# %%===========================================================================

import matplotlib.pylab as plt
import os
from pandas import DataFrame, read_csv
from histomicstk.annotations_and_masks.annotation_and_mask_utils import \
    _get_idxs_for_all_rois


GTCODE_PATH = os.path.join(
    '/home/mtageld/Desktop/HistomicsTK/histomicstk/annotations_and_masks/',
    'tests/test_files', 'sample_GTcodes.csv')

# just a temp directory to save masks for now
BASE_SAVEPATH = '/home/mtageld/Desktop/tmp/'
SAVEPATHS = {
    'ROI': os.path.join(BASE_SAVEPATH, 'masks'),
    'rgb': os.path.join(BASE_SAVEPATH, 'rgbs'),
    'contours': os.path.join(BASE_SAVEPATH, 'contours'),
    'visualization': os.path.join(BASE_SAVEPATH, 'vis'),
}
for _, savepath in SAVEPATHS.items():
    os.rmdir(savepath)
    os.mkdir(savepath)

# %%===========================================================================

# ---- PARAMETERS ----

gc = girder_client.GirderClient(apiUrl=APIURL)
gc.authenticate(apiKey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')

slide_id = SAMPLE_SLIDE_ID

GTCodes_dict = read_csv(GTCODE_PATH)
GTCodes_dict.index = GTCodes_dict.loc[:, 'group']
GTCodes_dict = GTCodes_dict.to_dict(orient='index')

save_directories = SAVEPATHS

annotations_to_contours_kwargs = {
    'MPP': 5.0, 'MAG': None,
    'linewidth': 0.2,
    'get_rgb': True, 'get_visualization': True,
}

slide_name = None
verbose = True
monitorprefix = ""

# %%===========================================================================

default_keyvalues = {
    'MPP': 5.0, 'MAG': None,
    'linewidth': 0.2,
    'get_rgb': True, 'get_visualization': True,
}

# assign defaults if nothing given
kvp = annotations_to_contours_kwargs  # for easy referencing
for k, v in default_keyvalues.items():
    if k not in kvp.keys():
        kvp[k] = v

# convert to df and sanity check
GTCodes_df = DataFrame.from_dict(GTCodes_dict, orient='index')
if any(GTCodes_df.loc[:, 'GT_code'] <= 0):
    raise Exception("All GT_code must be > 0")

# if not given, assign name of first file associated with girder item
if slide_name is None:
    resp = gc.get('/item/%s/files' % slide_id)
    slide_name = resp[0]['name']
    slide_name = slide_name[:slide_name.rfind('.')]

# get annotations for slide
slide_annotations = gc.get('/annotation/item/' + slide_id)

# scale up/down annotations by a factor
sf, _ = get_scale_factor_and_appendStr(
    gc=gc, slide_id=slide_id, MPP=kvp['MPP'], MAG=kvp['MAG'])
slide_annotations = scale_slide_annotations(slide_annotations, sf=sf)

# get bounding box information for all annotations
element_infos = get_bboxes_from_slide_annotations(slide_annotations)

# get idx of all 'special' roi annotations
idxs_for_all_rois = _get_idxs_for_all_rois(
    GTCodes=GTCodes_df, element_infos=element_infos)

savenames = []

for roino, idx_for_roi in enumerate(idxs_for_all_rois):

    roicountStr = "%s: roi %d of %d" % (
        monitorprefix, roino + 1, len(idxs_for_all_rois))

    # get specified area
    roi_out = annotations_to_contours_no_mask(
        gc=gc, slide_id=slide_id,
        mode='polygonal_bounds', idx_for_roi=idx_for_roi,
        slide_annotations=slide_annotations,
        element_infos=element_infos, **kvp)

    # now save roi (rgb, contours, vis)
    this_roi_savenames = dict()
    ROINAMESTR = "%s_left-%d_top-%d_bottom-%d_right-%d" % (
        slide_name,
        roi_out['bounds']['XMIN'], roi_out['bounds']['YMIN'],
        roi_out['bounds']['XMAX'], roi_out['bounds']['YMAX'])

    # TEMP !!!!!
    a

# %%===========================================================================

# Now for the contours to mask method
# This has
# - Two modes: semantic segmentation OR object segmentati
# - Two savetypes: png (max 255 objects, overlap lost)
#               OR pickle (no restrictions)
# The semantic segmentation mode ALWAYS saves pngs

contours = DataFrame(roi_out['contours'])
# GTCodes_df


# %%===========================================================================

# make sure ROI is overlayed first & assigned background class if relevant
roi_group = elinfos_roi.loc[idx_for_roi, 'group']
GTCodes_df.loc[roi_group, 'overlay_order'] = np.min(
    GTCodes_df.loc[:, 'overlay_order']) - 1
bck_classes = GTCodes_df.loc[
    GTCodes_df.loc[:, 'is_background_class'] == 1, :]
if bck_classes.shape[0] > 0:
    GTCodes_df.loc[
        roi_group, 'GT_code'] = bck_classes.iloc[0, :]['GT_code']

# Add annotations in overlay order
overlay_orders = sorted(set(GTCodes_df.loc[:, 'overlay_order']))
N_elements = elinfos_roi.shape[0]
elNo = 0
for overlay_level in overlay_orders:

    # get indices of relevant groups
    relevant_groups = list(GTCodes_df.loc[
        GTCodes_df.loc[:, 'overlay_order'] == overlay_level, 'group'])
    relIdxs = []
    for group_name in relevant_groups:
        relIdxs.extend(list(elinfos_roi.loc[
            elinfos_roi.group == group_name, :].index))

    # get relevnt infos and sort from largest to smallest (by bbox area)
    # so that the smaller elements are layered last. This helps partially
    # address issues describe in:
    # https://github.com/DigitalSlideArchive/HistomicsTK/issues/675
    elinfos_relevant = elinfos_roi.loc[relIdxs, :].copy()
    elinfos_relevant.sort_values(
        'bbox_area', axis=0, ascending=False, inplace=True)











