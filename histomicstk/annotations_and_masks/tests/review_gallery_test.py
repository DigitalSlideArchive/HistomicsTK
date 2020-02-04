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

from pandas import read_csv
from histomicstk.workflows.workflow_runner import (
    Workflow_runner, Slide_iterator)
from histomicstk.annotations_and_masks.annotation_and_mask_utils import \
    get_image_from_htk_response
from histomicstk.annotations_and_masks.annotations_to_masks_handler import \
    _visualize_annotations_on_rgb

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
    sld = gc.get('/item/%s' % slide_id)
    sldname = sld['name'][:sld['name'].find('.')]
    return get_all_rois_from_slide_v2(
        slide_id=slide_id, monitorprefix=monitorPrefix,
        # encoding slide id makes things easier later
        slide_name="%s_id-%s" % (sldname, slide_id),
        **kwargs)


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

sd = get_all_rois_kwargs['save_directories']
imnames = [
    j.replace('.png', '') for j in os.listdir(sd['rgb'])
    if j.endswith('.png')]
imnames.sort()

for imidx, imname in enumerate(imnames):

    # TEMP!!!!
    break

# %%===========================================================================

# how much smaller should the "zoomed out" rgb be
zoomout = 4

# %%===========================================================================

# read fetched rgb and visualization
rgb = imread(os.path.join(sd['rgb'], imname + '.png'))
vis = imread(os.path.join(sd['visualization'], imname + '.png'))
# conts = read_csv(
#     os.path.join(sd['contours'], imname + '.csv'))

slide_id = imname.split('_id-')[1].split('_')[0]

# get ROI location from imname
bounds = {
    locstr: int(imname.split(locstr + '-')[1].split('_')[0])
    for locstr in ('left', 'right', 'top', 'bottom')}

# get a lower-magnification surrounding field

# get append string for server request
getsf_kwargs = dict()
for k, v in get_all_rois_kwargs[
        'annotations_to_contours_kwargs'].items():
    if (k == 'MAG') and (v is not None):
        getsf_kwargs[k] = v / zoomout
    elif (k == 'MPP') and (v is not None):
        getsf_kwargs[k] = v * zoomout

sf, appendStr = get_scale_factor_and_appendStr(
    gc=gc, slide_id=slide_id, **getsf_kwargs)

# now get low-magnification surrounding field
x_margin = (bounds['right'] - bounds['left']) * zoomout / 2
y_margin = (bounds['bottom'] - bounds['top']) * zoomout / 2
getStr = \
    "/item/%s/tiles/region?left=%d&right=%d&top=%d&bottom=%d" \
    % (slide_id,
       bounds['left'] - x_margin,
       bounds['right'] + x_margin,
       bounds['top'] - y_margin,
       bounds['bottom'] + y_margin)
getStr += appendStr
resp = gc.get(getStr, jsonResp=False)
rgb_zoomout = get_image_from_htk_response(resp)

# plot a bounding box at the ROI region
xmin = x_margin * sf
xmax = xmin + (bounds['right'] - bounds['left']) * sf
ymin = y_margin * sf
ymax = ymin + (bounds['bottom'] - bounds['top']) * sf
xmin, xmax, ymin, ymax = [str(int(j)) for j in (xmin, xmax, ymin, ymax)]
contours_list = [{
    'color': 'rgb(255,0,0)',
    'coords_x': ",".join([xmin, xmax, xmax, xmin, xmin]),
    'coords_y': ",".join([ymin, ymin, ymax, ymax, ymin]),
}]
vis_zoomout = _visualize_annotations_on_rgb(rgb_zoomout, contours_list)

# %============================================================================

import io
from PIL import Image

# %============================================================================

wmax = max(vis.shape[1], vis_zoomout.shape[1])
hmax = max(vis.shape[0], vis_zoomout.shape[0])

fig, ax = plt.subplots(
    1, 3, dpi=100,
    figsize=(3 * wmax / 1000, hmax / 1000),
    gridspec_kw={'wspace': 0.01, 'hspace': 0}
)


ax[0].imshow(vis)
ax[1].imshow(rgb)
ax[2].imshow(vis_zoomout)

# plt.axis('off')
# ax = plt.gca()
#
# ax.set_xlim(0.0, rgb.shape[1])
# ax.set_ylim(0.0, rgb.shape[0])

for axis in ax:
    axis.axis('off')
fig.subplots_adjust(bottom=0, top=1, left=0, right=1)

buf = io.BytesIO()
plt.savefig(buf, format='png', pad_inches=0, dpi=1000)
buf.seek(0)
combined_vis = np.flipud(np.uint8(Image.open(buf))[..., :3])
plt.close()


plt.imshow(combined_vis)

