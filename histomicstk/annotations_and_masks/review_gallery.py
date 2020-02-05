import io
import matplotlib.pylab as plt
import numpy as np
from histomicstk.annotations_and_masks.annotations_to_object_mask_handler \
    import get_all_rois_from_slide_v2
from histomicstk.workflows.workflow_runner import Workflow_runner, \
    Slide_iterator


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


def _get_visualization_zoomout():
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
