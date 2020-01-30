from argparse import Namespace
from datetime import timedelta
from slicer_cli_web import ctk_cli_adjustment  # noqa - imported for side effects
from ctk_cli import CLIArgumentParser
import psutil
import numpy as np
import skimage.measure
import skimage.morphology

import histomicstk.preprocessing.color_deconvolution as htk_cdeconv
import histomicstk.segmentation as htk_seg
import histomicstk.utils as htk_utils

import large_image


# These defaults are only used if girder is not present
# Use memcached by default.
large_image.config.setConfig('cache_backend', 'memcached')
# If memcached is unavilable, specify the fraction of memory that python
# caching is allowed to use.  This is deliberately small.
large_image.config.setConfig('cache_python_memory_portion', 32)


def get_stain_vector(args, index):
    """Get the stain corresponding to args.stain_$index and
    args.stain_$index_vector.  If the former is not "custom", all the
    latter's elements must be -1.

    """
    args = args._asdict() if hasattr(args, '_asdict') else vars(args)
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


def segment_wsi_foreground_at_low_res(ts, lres_size=2048):

    ts_metadata = ts.getMetadata()

    # get image at low-res
    maxSize = max(ts_metadata['sizeX'], ts_metadata['sizeY'])
    maxSize = float(max(maxSize, lres_size))

    downsample_factor = 2.0 ** np.floor(np.log2(maxSize / lres_size))

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
            "fillColor": "rgba(0,0,0,0)",
            "lineColor": "rgb(0,255,0)"
        }

        nuclei_annot_list.append(cur_bbox)

    return nuclei_annot_list


def create_tile_nuclei_boundary_annotations(im_nuclei_seg_mask, tile_info):

    nuclei_annot_list = []

    gx = tile_info['gx']
    gy = tile_info['gy']
    wfrac = tile_info['gwidth'] / np.double(tile_info['width'])
    hfrac = tile_info['gheight'] / np.double(tile_info['height'])

    by, bx = htk_seg.label.trace_object_boundaries(im_nuclei_seg_mask,
                                                   trace_all=True)

    for i in range(len(bx)):

        # get boundary points and convert to base pixel space
        num_points = len(bx[i])

        if num_points < 3:
            continue

        cur_points = np.zeros((num_points, 3))
        cur_points[:, 0] = np.round(gx + bx[i] * wfrac, 2)
        cur_points[:, 1] = np.round(gy + by[i] * hfrac, 2)
        cur_points = cur_points.tolist()

        # create annotation json
        cur_annot = {
            "type": "polyline",
            "points": cur_points,
            "closed": True,
            "fillColor": "rgba(0,0,0,0)",
            "lineColor": "rgb(0,255,0)"
        }

        nuclei_annot_list.append(cur_annot)

    return nuclei_annot_list


def create_tile_nuclei_annotations(im_nuclei_seg_mask, tile_info, format):

    if format == 'bbox':

        return create_tile_nuclei_bbox_annotations(im_nuclei_seg_mask,
                                                   tile_info)

    elif format == 'boundary':

        return create_tile_nuclei_boundary_annotations(im_nuclei_seg_mask,
                                                       tile_info)
    else:

        raise ValueError('Invalid value passed for nuclei_annotation_format')


def create_dask_client(args):
    """Create and install a Dask distributed client using args from a
    Namespace, supporting the following attributes:

    - .scheduler: Address of the distributed scheduler, or the
      empty string to start one locally

    """
    import dask
    scheduler = getattr(args, 'scheduler', None)
    num_workers = getattr(args, 'num_workers', 0)
    num_threads_per_worker = getattr(args, 'num_threads_per_worker', 0)

    if scheduler == 'multithreading':
        import dask.threaded
        from multiprocessing.pool import ThreadPool

        if num_threads_per_worker <= 0:
            num_workers = max(1, psutil.cpu_count(logical=False) + num_threads_per_worker)
        else:
            num_workers = num_threads_per_worker
        print('Starting dask thread pool with %d thread(s)' % num_workers)
        dask.config.set(pool=ThreadPool(num_workers))
        dask.config.set(scheduler='threads')
        return

    if scheduler == 'multiprocessing':
        import dask.multiprocessing
        import multiprocessing

        dask.config.set(scheduler='processes')
        if num_workers <= 0:
            num_workers = max(1, psutil.cpu_count(logical=False) + num_workers)

        print('Starting dask multiprocessing pool with %d worker(s)' % num_workers)
        dask.config.set(pool=multiprocessing.Pool(
            num_workers, initializer=dask.multiprocessing.initialize_worker_process))
        return

    import dask.distributed
    if not scheduler:

        if num_workers <= 0:
            num_workers = max(1, psutil.cpu_count(logical=False) + num_workers)
        num_threads_per_worker = (num_threads_per_worker if num_threads_per_worker >= 1 else None)

        print('Creating dask LocalCluster with %d worker(s), %r thread(s) per '
              'worker' % (num_workers, num_threads_per_worker))
        scheduler = dask.distributed.LocalCluster(
            ip='0.0.0.0',  # Allow reaching the diagnostics port externally
            scheduler_port=0,  # Don't expose the scheduler port
            n_workers=num_workers,
            memory_limit=0,
            threads_per_worker=num_threads_per_worker,
            silence_logs=False
        )

    return dask.distributed.Client(scheduler)


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


def disp_time_hms(seconds):
    """Converts time from seconds to a string of the form hours:minutes:seconds
    """

    return str(timedelta(seconds=seconds))


def splitArgs(args, split='_'):
    """Split a Namespace into a Namespace of Namespaces based on shared
    prefixes.  The string separating the prefix from the rest of the
    argument is determined by the optional "split" parameter.
    Parameters not containing the splitting string are kept as-is.

    """
    def splitKey(k):
        s = k.split(split, 1)
        return (None, s[0]) if len(s) == 1 else s

    args = args._asdict() if hasattr(args, '_asdict') else vars(args)
    firstKeys = {splitKey(k)[0] for k in args}
    result = Namespace()
    for k in firstKeys - {None}:
        setattr(result, k, Namespace())
    for k, v in args.items():
        f, s = splitKey(k)
        if f is None:
            setattr(result, s, v)
        else:
            setattr(getattr(result, f), s, v)
    return result


def sample_pixels(args):
    """Version of histomicstk.utils.sample_pixels that takes a Namespace
    and handles the special default values.

    """
    args = (args._asdict() if hasattr(args, '_asdict') else vars(args)).copy()
    for k in 'magnification', 'sample_fraction', 'sample_approximate_total':
        if args[k] == -1:
            del args[k]
    return htk_utils.sample_pixels(**args)


__all__ = (
    'CLIArgumentParser',
    'create_dask_client',
    'create_tile_nuclei_annotations',
    'create_tile_nuclei_bbox_annotations',
    'create_tile_nuclei_boundary_annotations',
    'disp_time_hms',
    'get_region_dict',
    'get_stain_matrix',
    'get_stain_vector',
    'sample_pixels',
    'segment_wsi_foreground_at_low_res',
    'splitArgs',
)
