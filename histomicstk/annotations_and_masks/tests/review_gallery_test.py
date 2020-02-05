import copy
# import unittest

import os
# import tempfile
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
    _get_visualization_zoomout, _plot_rapid_review_vis

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

# where to save gallery
combinedvis_savepath = "/home/mtageld/Desktop/tmp/combinedvis/"
if os.path.exists(combinedvis_savepath):
    shutil.rmtree(combinedvis_savepath)
os.mkdir(combinedvis_savepath)

# %%===========================================================================
# %%===========================================================================

from histomicstk.annotations_and_masks.annotation_and_mask_utils import \
    get_image_from_htk_response
from histomicstk.annotations_and_masks.annotations_to_masks_handler import \
    _visualize_annotations_on_rgb
import io
from PIL import Image

from histomicstk.annotations_and_masks.review_gallery import \
    DTYPE_TO_FORMAT, FORMAT_TO_DTYPE

import pyvips

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
    'callback': _plot_rapid_review_vis,
    'callback_kwargs': {
        'combinedvis_savepath': combinedvis_savepath,
        'zoomout': 4,

    },
}

monitor = 'test'

# %%===========================================================================

# Get al rois to prep for gallery
get_all_rois_from_folder_v2(
    gc=gc, folderid=folderid, get_all_rois_kwargs=get_all_rois_kwargs,
    monitor=monitor)

# %%===========================================================================

# combinedvis_savepath
tilepath_base = combinedvis_savepath

# where to save gallery
gallery_savepath = "/home/mtageld/Desktop/tmp/gallery/"
if os.path.exists(gallery_savepath):
    shutil.rmtree(gallery_savepath)
os.mkdir(gallery_savepath)

padding = 25

tiles_per_row = 2
tiles_per_column = 5

# %%===========================================================================

tile_paths = [
    os.path.join(tilepath_base, j) for j in
    os.listdir(tilepath_base) if j.endswith('.png')]
tile_paths.sort()

n_tiles = len(tile_paths)
n_galleries = int(np.ceil(n_tiles / (tiles_per_row * tiles_per_column)))

tileidx = 0

for galno in range(n_galleries):

    # this makes a 8-bit, mono image (initializes as 1x1x3 matrix)
    im = pyvips.Image.black(1, 1, bands=3)

    for row in range(tiles_per_column):

        row_im = pyvips.Image.black(1, 1, bands=3)

        for col in range(tiles_per_row):

            if tileidx == n_tiles:
                break

            tilepath = tile_paths[tileidx]

            print("Inserting tile %d of %d: %s" % (
                tileidx, n_tiles, tilepath))
            tileidx += 1

            # # get tile from file
            tile = pyvips.Image.new_from_file(tilepath, access="sequential")

            # insert tile into mosaic row
            row_im = row_im.insert(
                tile[:3], row_im.width + padding, 0,
                expand=True, background=255)

        im = im.insert(
            row_im, 0, im.height + padding, expand=True, background=255)

    savepath = os.path.join(
        gallery_savepath, 'gallery-%d.tiff' % (galno + 1))
    print("Saving mosiac %d of %d to %s" % (galno + 1, n_galleries, savepath))
    im.tiffsave(
        savepath, tile=True, tile_width=256, tile_height=256, pyramid=True)

# %%===========================================================================
