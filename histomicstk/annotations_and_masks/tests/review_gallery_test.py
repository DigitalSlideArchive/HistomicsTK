import copy
# import unittest

import os
import tempfile
import shutil
import matplotlib.pylab as plt
from imageio import imread
from pandas import read_csv
import numpy as np
import girder_client
from histomicstk.annotations_and_masks.annotation_and_mask_utils import \
    get_scale_factor_and_appendStr
from histomicstk.annotations_and_masks.review_gallery import \
    get_all_rois_from_folder_v2, _get_review_visualization, \
    _get_visualization_zoomout, _plot_rapid_review_vis, create_review_galleries

# %%===========================================================================
# Constants & prep work


APIURL = 'http://candygram.neurology.emory.edu:8080/api/v1/'
SAMPLE_FOLDER_ID = '5e2a2da8ddda5f83986d18a2'

gc = girder_client.GirderClient(apiUrl=APIURL)
gc.authenticate(interactive=True)
# gc.authenticate(apiKey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')

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
# for _, savepath in SAVEPATHS.items():
#     if os.path.exists(savepath):
#         shutil.rmtree(savepath)
#     os.mkdir(savepath)

# where to save gallery
combinedvis_savepath = "/home/mtageld/Desktop/tmp/combinedvis/"
# if os.path.exists(combinedvis_savepath):
#     shutil.rmtree(combinedvis_savepath)
# os.mkdir(combinedvis_savepath)

# %%===========================================================================

# gc
folderid = SAMPLE_FOLDER_ID

monitor = 'test'

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
    'callback': _plot_rapid_review_vis,
    'callback_kwargs': {
        'combinedvis_savepath': combinedvis_savepath,
        'zoomout': 4,

    },
}

create_review_galleries_kwargs = {
    'tilepath_base': combinedvis_savepath,
    'upload_results': True,
    'gc': gc,
    'gallery_folderid': "5e1e854e90ef76f6ac8c43bd",
    'gallery_savepath': None,
    'padding': 25,
    'tiles_per_row': 2,
    'tiles_per_column': 5,
}

# %%===========================================================================

# Get al rois to prep for gallery
# get_all_rois_from_folder_v2(
#     gc=gc, folderid=folderid, get_all_rois_kwargs=get_all_rois_kwargs,
#     monitor=monitor)

# create (+/- post) review gallery
resps = create_review_galleries(**create_review_galleries_kwargs)

# %%===========================================================================
