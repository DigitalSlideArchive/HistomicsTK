import io
import os
import tempfile

import numpy as np
import pyvips
from imageio import imwrite
from PIL import Image

from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    get_image_from_htk_response, get_scale_factor_and_appendStr)
from histomicstk.annotations_and_masks.annotations_to_masks_handler import \
    _visualize_annotations_on_rgb
from histomicstk.annotations_and_masks.annotations_to_object_mask_handler import \
    get_all_rois_from_slide_v2
from histomicstk.annotations_and_masks.masks_to_annotations_handler import \
    get_annotation_documents_from_contours
from histomicstk.workflows.workflow_runner import (Slide_iterator,
                                                   Workflow_runner)

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


def get_all_rois_from_folder_v2(
        gc, folderid, get_all_rois_kwargs, monitor=''):
    """Get all rois in a girder folder using get_all_rois_from_slide_v2().

    Parameters
    ----------
    gc : girder_client.Girder_Client
        authenticated girder client
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
        if '.' not in sld['name']:
            sld['name'] += '.'
        sldname = sld['name'][:sld['name'].find('.')].replace('/', '_#_')
        return get_all_rois_from_slide_v2(
            slide_id=slide_id, monitorprefix=monitorPrefix,
            # encoding slide id makes things easier later
            slide_name='%s_id-%s' % (sldname, slide_id),
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
        monitorPrefix=monitor,
    )
    workflow_runner.run()


def _get_visualization_zoomout(
        gc, slide_id, bounds, MPP, MAG, zoomout=4):
    """Get a zoomed out visualization of ROI RGB and annotation overlay.

    Parameters
    ----------
    gc : girder_client.Girder_Client
        authenticated girder client
    slide_id : str
        girder ID of slide
    bounds : dict
        bounds of the region of interest. Must contain the keys
        XMIN, XMAX, YMIN, YMAX
    MPP : float
        Microns per pixel.
    MAG : float
        Magnification. MPP overrides this.
    zoomout : float
        how much to zoom out

    Returns
    -------
    np.array
        Zoomed out visualization. Output from _visualize_annotations_on_rgb().

    """
    # get append string for server request
    if MPP is not None:
        getsf_kwargs = {
            'MPP': MPP * (zoomout + 1),
            'MAG': None,
        }
    elif MAG is not None:
        getsf_kwargs = {
            'MPP': None,
            'MAG': MAG / (zoomout + 1),
        }
    else:
        getsf_kwargs = {
            'MPP': None,
            'MAG': None,
        }
    sf, appendStr = get_scale_factor_and_appendStr(
        gc=gc, slide_id=slide_id, **getsf_kwargs)

    # now get low-magnification surrounding field
    x_margin = (bounds['XMAX'] - bounds['XMIN']) * zoomout / 2
    y_margin = (bounds['YMAX'] - bounds['YMIN']) * zoomout / 2
    getStr = \
        '/item/%s/tiles/region?left=%d&right=%d&top=%d&bottom=%d' \
        % (slide_id,
           max(0, bounds['XMIN'] - x_margin),
           bounds['XMAX'] + x_margin,
           max(0, bounds['YMIN'] - y_margin),
           bounds['YMAX'] + y_margin)
    getStr += appendStr
    resp = gc.get(getStr, jsonResp=False)
    rgb_zoomout = get_image_from_htk_response(resp)

    # plot a bounding box at the ROI region
    xmin = x_margin * sf
    xmax = xmin + (bounds['XMAX'] - bounds['XMIN']) * sf
    ymin = y_margin * sf
    ymax = ymin + (bounds['YMAX'] - bounds['YMIN']) * sf
    xmin, xmax, ymin, ymax = (str(int(j)) for j in (xmin, xmax, ymin, ymax))
    contours_list = [{
        'color': 'rgb(255,255,0)',
        'coords_x': ','.join([xmin, xmax, xmax, xmin, xmin]),
        'coords_y': ','.join([ymin, ymin, ymax, ymax, ymin]),
    }]

    return _visualize_annotations_on_rgb(rgb_zoomout, contours_list)


def _get_review_visualization(rgb, vis, vis_zoomout):
    """Get a visualization of rgb and annotations for rapid review.

    Parameters
    ----------
    rgb : np.array
        mxnx3 rgb image
    vis : np.array
        visualization of rgb with overlaid annotations
    vis_zoomout
        same as vis, but at a lower magnififcation.

    Returns
    -------
    np.array
        visualization to be used for gallery

    """
    import matplotlib.pyplot as plt

    wmax = max(vis.shape[1], vis_zoomout.shape[1])
    hmax = max(vis.shape[0], vis_zoomout.shape[0])

    fig, ax = plt.subplots(
        1, 3, dpi=100,
        figsize=(3 * wmax / 1000, hmax / 1000),
        gridspec_kw={'wspace': 0.01, 'hspace': 0},
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
    combined_vis = np.uint8(Image.open(buf))[..., :3]
    plt.close()

    return combined_vis


def _plot_rapid_review_vis(
        roi_out, gc, slide_id, slide_name, MPP, MAG,
        combinedvis_savepath, zoomout=4,
        verbose=False, monitorprefix=''):
    """Plot a visualization for rapid review of ROI.

    This is a callback to be called inside get_all_rois_from_slide_v2().

    Parameters
    ----------
    roi_out : dict
        output from annotations_to_contours_no_mask()
    gc : girder_client.Girder_Client
        authenticated girder client
    slide_id : str
        girder slide id
    slide_name : str
        name of the slide
    MPP : float
        microns per pixel
    MAG : float
        magnification. superseded by MPP.
    combinedvis_savepath : str
        path to save the combined visualization
    zoomout : float
        how much to zoom out to get the gallery visualization
    verbose : bool
        print statements to screen
    monitorprefix : str
        text to prepent to printed statements

    Returns
    -------
    dict
        roi_out parameter whether or not it is modified

    """
    # get rgb and visualization (fetched mag + lower mag)
    vis_zoomout = _get_visualization_zoomout(
        gc=gc, slide_id=slide_id, bounds=roi_out['bounds'],
        MPP=MPP, MAG=MAG, zoomout=zoomout)

    # combined everything in a neat visualization for rapid review
    ROINAMESTR = '%s_left-%d_top-%d_bottom-%d_right-%d' % (
        slide_name,
        roi_out['bounds']['XMIN'], roi_out['bounds']['YMIN'],
        roi_out['bounds']['YMAX'], roi_out['bounds']['XMAX'])
    savename = os.path.join(combinedvis_savepath, ROINAMESTR + '.png')
    rapid_review_vis = _get_review_visualization(
        rgb=roi_out['rgb'], vis=roi_out['visualization'],
        vis_zoomout=vis_zoomout)

    # save visualization for later use
    if verbose:
        print('%s: Saving %s' % (monitorprefix, savename))
    imwrite(im=rapid_review_vis, uri=savename)

    return roi_out


def create_review_galleries(
        tilepath_base, upload_results=True, gc=None,
        gallery_savepath=None, gallery_folderid=None,
        padding=25, tiles_per_row=2, tiles_per_column=5,
        annprops=None, url=None, nameprefix=''):
    """Create and or post review galleries for rapid review.

    Parameters
    ----------
    tilepath_base : str
        directory where combined visualization.
    upload_results : bool
        upload results to DSA?
    gc : girder_client.Girder_Client
        authenticated girder client. Only needed upload_results.
    gallery_savepath : str
        directory to save gallery. Only if upload_results.
    gallery_folderid : str
        girder ID of folder to post galleries. Only if upload_result.
    padding : int
        padding in pixels between tiles in same gallery.
    tiles_per_row : int
        how many visualization tiles per row in gallery.
    tiles_per_column : int
        how many visualization tiles per column in gallery.
    annprops : dict
        properties of the annotations to be posted to DSA. Passed directly
        as annprops to get_annotation_documents_from_contours()
    url : str
        url of the Digital Slide Archive Instance. For example:
        http://candygram.neurology.emory.edu:8080/
    nameprefix : str
        prefix to prepend to gallery name

    Returns
    -------
    list
        each entry is a dict representing the response of the server
        post request to upload the gallery to DSA.

    """
    from pandas import DataFrame

    if upload_results:
        for par in ('gc', 'gallery_folderid', 'url'):
            if locals()[par] is None:
                raise Exception(
                    '%s cannot be None if upload_results!' % par)

    if gallery_savepath is None:
        gallery_savepath = tempfile.mkdtemp(prefix='gallery-')

    savepaths = []
    resps = []

    tile_paths = sorted([
        os.path.join(tilepath_base, j) for j in
        os.listdir(tilepath_base) if j.endswith('.png')])

    def _parse_tilepath(tpath):
        basename = os.path.basename(tpath)
        basename = basename[:basename.rfind('.')]
        tileinfo = {'slide_name': basename.split('_')[0]}
        for attrib in ['id', 'left', 'top', 'bottom', 'right']:
            tileinfo[attrib] = basename.split(
                attrib + '-')[1].split('_')[0]

        # add URL in histomicsTK
        tileinfo['URL'] = url + \
            'histomicstk#?image=%s&bounds=%s%%2C%s%%2C%s%%2C%s%%2C0' % (
                tileinfo['id'],
                tileinfo['left'], tileinfo['top'],
                tileinfo['right'], tileinfo['bottom'])
        return tileinfo

    n_tiles = len(tile_paths)
    n_galleries = int(np.ceil(n_tiles / (tiles_per_row * tiles_per_column)))

    tileidx = 0

    for galno in range(n_galleries):

        # this makes a 8-bit, mono image (initializes as 1x1x3 matrix)
        im = pyvips.Image.black(1, 1, bands=3)

        # this will store the roi contours
        contours = []

        for _row in range(tiles_per_column):

            rowpos = im.height + padding

            # initialize "row" strip image
            row_im = pyvips.Image.black(1, 1, bands=3)

            for _col in range(tiles_per_row):

                if tileidx == n_tiles:
                    break

                tilepath = tile_paths[tileidx]

                print('Inserting tile %d of %d: %s' % (
                    tileidx, n_tiles, tilepath))
                tileidx += 1

                # # get tile from file
                tile = pyvips.Image.new_from_file(
                    tilepath, access='sequential')

                # insert tile into mosaic row
                colpos = row_im.width + padding
                row_im = row_im.insert(
                    tile[:3], colpos, 0, expand=True, background=255)

                if upload_results:

                    tileinfo = _parse_tilepath(tilepath)

                    xmin = colpos
                    ymin = rowpos
                    xmax = xmin + tile.width
                    ymax = ymin + tile.height
                    xmin, xmax, ymin, ymax = (
                        str(j) for j in (xmin, xmax, ymin, ymax))
                    contours.append({
                        'group': tileinfo['slide_name'],
                        'label': tileinfo['URL'],
                        'color': 'rgb(0,0,0)',
                        'coords_x': ','.join([xmin, xmax, xmax, xmin, xmin]),
                        'coords_y': ','.join([ymin, ymin, ymax, ymax, ymin]),
                    })

                    # Add a small contour so that when the pathologist
                    # changes the label to approve or disapprove of the
                    # FOV, the URL in THIS contour (a link to the original
                    # FOV) can be used. We place it in the top right corner.
                    boxsize = 25
                    xmin = str(int(xmax) - boxsize)
                    ymax = str(int(ymin) + boxsize)
                    contours.append({
                        'group': tileinfo['slide_name'],
                        'label': tileinfo['URL'],
                        'color': 'rgb(0,0,0)',
                        'coords_x': ','.join([xmin, xmax, xmax, xmin, xmin]),
                        'coords_y': ','.join([ymin, ymin, ymax, ymax, ymin]),
                    })

            # insert row into main gallery
            im = im.insert(row_im, 0, rowpos, expand=True, background=255)

        filename = '%s_gallery-%d' % (nameprefix, galno + 1)
        savepath = os.path.join(gallery_savepath, filename + '.tiff')
        print('Saving gallery %d of %d to %s' % (
            galno + 1, n_galleries, savepath))

        # save temporarily to disk to be uploaded
        im.tiffsave(
            savepath, tile=True, tile_width=256, tile_height=256, pyramid=True)

        if upload_results:
            # upload the gallery to DSA
            resps.append(gc.uploadFileToFolder(
                folderId=gallery_folderid, filepath=savepath,
                filename=filename))
            os.remove(savepath)

            # get and post FOV location annotations
            annotation_docs = get_annotation_documents_from_contours(
                DataFrame(contours), separate_docs_by_group=True,
                annprops=annprops)
            for doc in annotation_docs:
                _ = gc.post(
                    '/annotation?itemId=' + resps[-1]['itemId'], json=doc)
        else:
            savepaths.append(savepath)

    return resps if upload_results else savepaths
