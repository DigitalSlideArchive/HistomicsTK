import unittest

import os
import girder_client
from pandas import read_csv
import tempfile
import shutil

from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    get_bboxes_from_slide_annotations, _get_idxs_for_all_rois,
    scale_slide_annotations, get_scale_factor_and_appendStr)

# %%===========================================================================
# Constants & prep work

APIURL = 'http://candygram.neurology.emory.edu:8080/api/v1/'
SAMPLE_SLIDE_ID = '5d586d57bd4404c6b1f28640'

gc = girder_client.GirderClient(apiUrl=APIURL)
# gc.authenticate(interactive=True)
gc.authenticate(apiKey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')

# just a temp directory to save masks for now
# BASE_SAVEPATH = tempfile.mkdtemp()
BASE_SAVEPATH = "/home/mtageld/Desktop/tmp/"
SAVEPATHS = {
    'rgb': os.path.join(BASE_SAVEPATH, 'rgbs'),
    'contours': os.path.join(BASE_SAVEPATH, 'contours'),
    'visualization': os.path.join(BASE_SAVEPATH, 'vis'),
}
for _, savepath in SAVEPATHS.items():
    os.mkdir(savepath)

# Microns-per-pixel / Magnification (either or)
MPP = 5.0
MAG = None

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

# params for get_annotations_in_slide_region()
get_kwargs = {
    'gc': gc, 'slide_id': SAMPLE_SLIDE_ID,
    'bounds': {
        'XMIN': 58000, 'XMAX': 63000,
        'YMIN': 35000, 'YMAX': 39000},
    'MPP': MPP,
    'MAG': MAG,
    'get_rgb': True,
    'get_visualization': True,
}

# %%===========================================================================









