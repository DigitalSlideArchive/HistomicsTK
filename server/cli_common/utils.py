import numpy as np
import scipy as sp
import skimage.measure
import multiprocessing
import dask.distributed

import histomicstk.preprocessing.color_deconvolution as htk_cdeconv
import histomicstk.filters.shape as htk_shape_filters
import histomicstk.segmentation as htk_seg
import histomicstk.utils as htk_utils

import large_image


def get_stain_vector(args, index):
    """Get the stain corresponding to args.stain_$index and
    args.stain_$index_vector.  If the former is not "custom", all the
    latter's elements must be -1.

    """
    args = vars(args)
    stain = args['stain_' + str(index)]
    stain_vector = args['stain_' + str(index) + '_vector']
    if all(x == -1 for x in stain_vector):  # Magic default value
        if stain == 'custom':
            raise ValueError('If "custom" is chosen for a stain, '
                             'a stain vector must be provided.')
        return htk_cdeconv.stain_color_map[stain]
    else:
        if stain == 'custom':
            return stain_vector
        raise ValueError('Unless "custom" is chosen for a stain, '
                         'no stain vector may be provided.')


def get_stain_matrix(args, count=3):
    """Get the stain matrix corresponding to the args.stain_$index and
    args.stain_$index_vector arguments for values of index 1 to count.
    Return a numpy array of column vectors.

    """
    return np.array([get_stain_vector(args, i+1) for i in range(count)]).T


def segment_wsi_foreground_at_low_res(ts):

    ts_metadata = ts.getMetadata()

    # get image at low-res
    maxSize = max(ts_metadata['sizeX'], ts_metadata['sizeY'])

    downsample_factor = 2 ** np.floor(np.log2(maxSize / 2048))

    fgnd_seg_mag = ts_metadata['magnification'] / downsample_factor

    fgnd_seg_scale = {'magnification': fgnd_seg_mag}

    im_lres, _ = ts.getRegion(
        scale=fgnd_seg_scale,
        format=large_image.tilesource.TILE_FORMAT_NUMPY
    )

    im_lres = im_lres[:, :, :3]

    # compute foreground mask at low-res
    im_fgnd_mask_lres = htk_utils.simple_mask(im_lres)

    return im_fgnd_mask_lres, fgnd_seg_scale


def detect_nuclei_kofahi(im_nuclei_stain, args):

    # segment foreground (assumes nuclei are darker on a bright background)
    im_nuclei_fgnd_mask = sp.ndimage.morphology.binary_fill_holes(
        im_nuclei_stain < args.foreground_threshold)

    # run adaptive multi-scale LoG filter
    im_log_max, im_sigma_max = htk_shape_filters.cdog(
        im_nuclei_stain, im_nuclei_fgnd_mask,
        sigma_min=args.min_radius / np.sqrt(2),
        sigma_max=args.max_radius / np.sqrt(2)
    )

    # apply local maximum clustering
    im_nuclei_seg_mask, seeds, maxima = htk_seg.nuclear.max_clustering(
        im_log_max, im_nuclei_fgnd_mask, args.local_max_search_radius)

    # filter out small objects
    im_nuclei_seg_mask = htk_seg.label.area_open(
        im_nuclei_seg_mask, args.min_nucleus_area).astype(np.int)

    return im_nuclei_seg_mask


def create_tile_nuclei_bbox_annotations(im_nuclei_seg_mask, tile_info):

    nuclei_annot_list = []

    gx = tile_info['gx']
    gy = tile_info['gy']
    wfrac = tile_info['gwidth'] / np.double(tile_info['width'])
    hfrac = tile_info['gheight'] / np.double(tile_info['height'])

    nuclei_obj_props = skimage.measure.regionprops(im_nuclei_seg_mask)

    for i in range(len(nuclei_obj_props)):
        cx = nuclei_obj_props[i].centroid[1]
        cy = nuclei_obj_props[i].centroid[0]
        width = nuclei_obj_props[i].bbox[3] - nuclei_obj_props[i].bbox[1] + 1
        height = nuclei_obj_props[i].bbox[2] - nuclei_obj_props[i].bbox[0] + 1

        # convert to base pixel coords
        cx = np.round(gx + cx * wfrac, 2)
        cy = np.round(gy + cy * hfrac, 2)
        width = np.round(width * wfrac, 2)
        height = np.round(height * hfrac, 2)

        # create annotation json
        cur_bbox = {
            "type": "rectangle",
            "center": [cx, cy, 0],
            "width": width,
            "height": height,
            "rotation": 0,
            "fillColor": "rgba(0,0,0,0)"
        }

        nuclei_annot_list.append(cur_bbox)

    return nuclei_annot_list


def create_tile_nuclei_boundary_annotations(im_nuclei_seg_mask, tile_info):

    nuclei_annot_list = []

    gx = tile_info['gx']
    gy = tile_info['gy']
    wfrac = tile_info['gwidth'] / np.double(tile_info['width'])
    hfrac = tile_info['gheight'] / np.double(tile_info['height'])

    bx, by = htk_seg.label.trace_object_boundaries(im_nuclei_seg_mask,
                                                   trace_all=True)

    for i in range(len(bx)):

        # get boundary points and convert to base pixel space
        num_points = len(bx[i])

        cur_points = np.zeros((num_points, 3))
        cur_points[:, 0] = np.round(gx + bx[i] * wfrac, 2)
        cur_points[:, 1] = np.round(gy + by[i] * hfrac, 2)

        # create annotation json
        cur_annot = {
            "type": "polyline",
            "points": cur_points.tolist(),
            "closed": True,
            "fillColor": "rgba(0,0,0,0)"
        }

        nuclei_annot_list.append(cur_annot)

    return nuclei_annot_list


def create_tile_nuclei_annotations(im_nuclei_seg_mask, tile_info, args):

    if args.nuclei_annotation_format == 'bbox':

        return create_tile_nuclei_bbox_annotations(im_nuclei_seg_mask,
                                                   tile_info)

    elif args.nuclei_annotation_format == 'boundary':

        return create_tile_nuclei_boundary_annotations(im_nuclei_seg_mask,
                                                       tile_info)
    else:

        raise ValueError('Invalid value passed for nuclei_annotation_format')


def create_dask_client(args):

    scheduler_address = args.scheduler_address

    if not scheduler_address:

        scheduler_address = dask.distributed.LocalCluster(
            n_workers=multiprocessing.cpu_count()-1,
            scheduler_port=0,
            silence_logs=False
        )

    c = dask.distributed.Client(scheduler_address)

    return c


def get_region_dict(region, maxRegionSize=None, tilesource=None):
    """Return a dict corresponding to region, checking the region size if
    maxRegionSize is provided.

    The intended use is to be passed via **kwargs, and so either {} is
    returned (for the special region -1,-1,-1,-1) or {'region':
    region_dict}.

    Params
    ------
    region: list
        4 elements -- left, top, width, height -- or all -1, meaning the whole
        slide.
    maxRegionSize: int, optional
        Maximum size permitted of any single dimension
    tilesource: tilesource, optional
        A `large_image` tilesource (or anything with `.sizeX` and `.sizeY`
        properties) that is used to determine the size of the whole slide if
        necessary.  Must be provided if `maxRegionSize` is.

    Returns
    -------
    region_dict: dict
        Either {} (for the special region -1,-1,-1,-1) or
        {'region': region_subdict}

    """

    if len(region) != 4:
        raise ValueError('Exactly four values required for --region')

    useWholeImage = region == [-1] * 4

    if maxRegionSize is not None:
        if tilesource is None:
            raise ValueError('tilesource must be provided if maxRegionSize is')
        if maxRegionSize != -1:
            if useWholeImage:
                size = max(tilesource.sizeX, tilesource.sizeY)
            else:
                size = max(region[-2:])
            if size > maxRegionSize:
                raise ValueError('Requested region is too large!  '
                                 'Please see --maxRegionSize')

    return {} if useWholeImage else dict(
        region=dict(zip(['left', 'top', 'width', 'height'],
                        region)))


__all__ = (
    'create_dask_client',
    'create_tile_nuclei_annotations',
    'create_tile_nuclei_bbox_annotations',
    'create_tile_nuclei_boundary_annotations',
    'detect_nuclei_kofahi',
    'get_region_dict',
    'get_stain_matrix',
    'get_stain_vector',
    'segment_wsi_foreground_at_low_res',
)
