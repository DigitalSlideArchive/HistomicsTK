import copy
# import unittest

import os
# import tempfile
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
SAMPLE_FOLDER_ID = '5e2a2da8ddda5f83986d18a2'

gc = girder_client.GirderClient(apiUrl=APIURL)
# gc.authenticate(interactive=True)
gc.authenticate(apiKey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')

# GT codes dict for parsing into label mask
GTCODE_PATH = os.path.join(
    '/home/mtageld/Desktop/HistomicsTK/histomicstk/annotations_and_masks/',
    'tests/test_files', 'sample_GTcodes_v2.csv')
GTCodes_dict = read_csv(GTCODE_PATH)
GTCodes_dict.index = GTCodes_dict.loc[:, 'group']
GTCodes_dict = GTCodes_dict.to_dict(orient='index')

# just a temp directory to save masks for now
# BASE_SAVEPATH = tempfile.mkdtemp()
BASE_SAVEPATH = '/home/mtageld/Desktop/tmp/'
SAVEPATHS = {
    'contours': os.path.join(BASE_SAVEPATH, 'contours'),
    'rgb': os.path.join(BASE_SAVEPATH, 'rgbs'),
    'visualization': os.path.join(BASE_SAVEPATH, 'vis'),
}
for _, savepath in SAVEPATHS.items():
    if os.path.exists(savepath):
        shutil.rmtree(savepath)
    os.mkdir(savepath)

# %%===========================================================================
# %%===========================================================================

from histomicstk.workflows.workflow_runner import (
    Workflow_runner, Slide_iterator)

# %%===========================================================================

# gc
folderid = SAMPLE_FOLDER_ID

# params for getting all rois for slide
get_all_rois_kwargs = {
    'GTCodes_dict': GTCodes_dict,
    'save_directories': SAVEPATHS,
    'annotations_to_contours_kwargs': {
        'MPP': 0.2,
        'linewidth': 0.2,
        'get_rgb': True,
        'get_visualization': True,
    },
    'verbose': False,
    'get_mask': False,
}

monitor = 'test'

# %%===========================================================================


def _get_all_rois(slide_id, monitorPrefix, **kwargs):
    return get_all_rois_from_slide_v2(
        slide_id=slide_id, monitorprefix=monitorPrefix, **kwargs)


# update with params
get_all_rois_kwargs['gc'] = gc

# pull annotations for each slide in folder
workflow_runner = Workflow_runner(
    slide_iterator=Slide_iterator(
        gc, source_folder_id=folderid,
        keep_slides=None,
    ),
    workflow=_get_all_rois,
    workflow_kwargs=get_all_rois_kwargs,
    monitorPrefix=monitor
)

workflow_runner.run()

# %%===========================================================================























