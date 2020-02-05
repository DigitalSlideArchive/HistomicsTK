import io
import os

import matplotlib.pylab as plt
import numpy as np
from PIL import Image
from imageio import imwrite

from histomicstk.annotations_and_masks.annotation_and_mask_utils import \
    get_scale_factor_and_appendStr, get_image_from_htk_response
from histomicstk.annotations_and_masks.annotations_to_masks_handler import \
    _visualize_annotations_on_rgb
from histomicstk.annotations_and_masks.annotations_to_object_mask_handler \
    import get_all_rois_from_slide_v2
from histomicstk.workflows.workflow_runner import Workflow_runner, \
    Slide_iterator

# %============================================================================
# CONSTANTS

# source: https://libvips.github.io/libvips/API/current/Examples.md.html
# source 2: https://libvips.github.io/libvips/API/current/Examples.md.html
# source 3: https://github.com/libvips/pyvips/issues/109
# source 4: https://github.com/libvips/libvips/issues/1254

# map np dtypes to vips
DTYPE_TO_FORMAT = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}

# map vips formats to np dtypes
FORMAT_TO_DTYPE = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

# %============================================================================


def get_all_rois_from_folder_v2(
        gc, folderid, get_all_rois_kwargs, monitor=''):
    """Get all rois in a girder folder using get_all_rois_from_slide_v2().

    Parameters
    ----------
    gc : girder_client.Girder_Client
        connected girder client
    folderid : str
        girder id of folder
    get_all_rois_kwargs : dict
        kwargs to pass to get_all_rois_from_slide_v2()
    monitor : str
        monitor prefix

    Returns
    -------
    None

    """
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


def _get_visualization_zoomout(
        gc, slide_id, bounds, MPP, MAG, zoomout=4):
    """Get a zoomed out visualization of ROI RGB and annotation overlay.

    Parameters
    ----------
    gc : girder_client.Girder_Client
        connected girder client
    zoomout : float
        how much to zoom out

    Returns
    -------

    """
    # get append string for server request
    getsf_kwargs = dict()
    if MPP is not None:
        getsf_kwargs = {
            'MPP': MPP * zoomout,
            'MAG': None,
        }
    else:
        getsf_kwargs = {
            'MPP': None,
            'MAG': MAG / zoomout,
        }
    sf, appendStr = get_scale_factor_and_appendStr(
        gc=gc, slide_id=slide_id, **getsf_kwargs)

    # now get low-magnification surrounding field
    x_margin = (bounds['XMAX'] - bounds['XMIN']) * zoomout / 2
    y_margin = (bounds['YMAX'] - bounds['YMIN']) * zoomout / 2
    getStr = \
        "/item/%s/tiles/region?left=%d&right=%d&top=%d&bottom=%d" \
        % (slide_id,
           bounds['XMIN'] - x_margin,
           bounds['XMAX'] + x_margin,
           bounds['YMIN'] - y_margin,
           bounds['YMAX'] + y_margin)
    getStr += appendStr
    resp = gc.get(getStr, jsonResp=False)
    rgb_zoomout = get_image_from_htk_response(resp)

    # plot a bounding box at the ROI region
    xmin = x_margin * sf
    xmax = xmin + (bounds['XMAX'] - bounds['XMIN']) * sf
    ymin = y_margin * sf
    ymax = ymin + (bounds['YMAX'] - bounds['YMIN']) * sf
    xmin, xmax, ymin, ymax = [str(int(j)) for j in (xmin, xmax, ymin, ymax)]
    contours_list = [{
        'color': 'rgb(255,0,0)',
        'coords_x': ",".join([xmin, xmax, xmax, xmin, xmin]),
        'coords_y': ",".join([ymin, ymin, ymax, ymax, ymin]),
    }]

    return _visualize_annotations_on_rgb(rgb_zoomout, contours_list)


def _get_review_visualization(rgb, vis, vis_zoomout):
    """Get a visualization of rgb and annotations for rapid review.

    Parameters
    ----------
    rgb : np.array
        mxnx3 rgb image
    vis : np.array
        visualization of rgb with overlayed annotations
    vis_zoomout
        same as vis, but at a lower magnififcation.

    Returns
    -------
    np.array
        visualization to be used for galler

    """
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

    for axis in ax:
        axis.axis('off')
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', pad_inches=0, dpi=1000)
    buf.seek(0)
    combined_vis = np.flipud(np.uint8(Image.open(buf))[..., :3])
    plt.close()

    return combined_vis


def _plot_rapid_review_vis(
        roi_out, gc, slide_id, slide_name, MPP, MAG,
        gallery_savepath, zoomout=4,
        verbose=False, monitorprefix=''):
    """Plot a visualization for rapid review of ROI.

    This is a callback to be called inside get_all_rois_from_slide_v2().

    Parameters
    ----------
    roi_out
    gc
    slide_id
    slide_name
    MPP
    MAG
    gallery_savepath
    zoomout
    verbose
    monitorprefix

    Returns
    -------

    """
    # get rgb and visualization (fetched mag + lower mag)
    vis_zoomout = _get_visualization_zoomout(
        gc=gc, slide_id=slide_id, bounds=roi_out['bounds'],
        MPP=MPP, MAG=MAG, zoomout=zoomout)

    # combined everything in a neat visualization for rapid review
    ROINAMESTR = "%s_left-%d_top-%d_bottom-%d_right-%d" % (
        slide_name,
        roi_out['bounds']['XMIN'], roi_out['bounds']['YMIN'],
        roi_out['bounds']['YMAX'], roi_out['bounds']['XMAX'])
    savename = os.path.join(gallery_savepath, ROINAMESTR + ".png")
    rapid_review_vis = _get_review_visualization(
        rgb=roi_out['rgb'], vis=roi_out['visualization'],
        vis_zoomout=vis_zoomout)

    # save visualization for later use
    if verbose:
        print("%s: Saving %s" % (monitorprefix, savename))
    imwrite(im=rapid_review_vis, uri=savename)

# %============================================================================
