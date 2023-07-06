from argparse import Namespace
from datetime import timedelta

import large_image
import numpy as np
from ctk_cli import CLIArgumentParser  # noqa I004
# imported for side effects
from slicer_cli_web import ctk_cli_adjustment  # noqa

import histomicstk.preprocessing.color_deconvolution as htk_cdeconv
import histomicstk.segmentation as htk_seg
import histomicstk.utils as htk_utils

# These defaults are only used if girder is not present
# Use memcached by default.
large_image.config.setConfig('cache_backend', 'memcached')
# If memcached is unavailable, specify the fraction of memory that python
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
    return np.array([get_stain_vector(args, i + 1) for i in range(count)]).T


def segment_wsi_foreground_at_low_res(
        ts, lres_size=2048, invert_image=False, frame=None, default_img_inversion=False):

    ts_metadata = ts.getMetadata()

    # get image at low-res
    maxSize = max(ts_metadata['sizeX'], ts_metadata['sizeY'])
    maxSize = float(max(maxSize, lres_size))

    downsample_factor = 2.0 ** np.floor(np.log2(maxSize / lres_size))

    fgnd_seg_mag = ts_metadata['magnification'] / downsample_factor

    fgnd_seg_scale = {'magnification': fgnd_seg_mag}

    im_lres, _ = ts.getRegion(
        scale=fgnd_seg_scale,
        format=large_image.tilesource.TILE_FORMAT_NUMPY,
        frame=frame
    )

    # check number of channels
    if len(im_lres.shape) <= 2 or im_lres.shape[2] == 1:
        im_lres = np.dstack((im_lres, im_lres, im_lres))
        if default_img_inversion:
            invert_image = True
    else:
        im_lres = im_lres[:, :, :3]

    # perform image inversion
    if invert_image:
        im_lres = np.max(im_lres) - im_lres

    # casting the float to integers
    if issubclass(im_lres.dtype.type, np.floating) and np.max(im_lres) > 1:
        if np.min(im_lres) >= 0 and np.max(im_lres) < 256:
            im_lres = im_lres.astype(np.uint8)
        elif np.min(im_lres) >= 0 and np.max(im_lres) < 65536:
            im_lres = im_lres.astype(np.uint16)

    # compute foreground mask at low-res
    im_fgnd_mask_lres = htk_utils.simple_mask(im_lres)

    return im_fgnd_mask_lres, fgnd_seg_scale


def create_tile_nuclei_bbox_annotations(im_nuclei_seg_mask, tile_info):
    import skimage.measure

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
            'type': 'rectangle',
            'center': [cx, cy, 0],
            'width': width,
            'height': height,
            'rotation': 0,
            'fillColor': 'rgba(0,0,0,0)',
            'lineColor': 'rgb(0,255,0)'
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
            'type': 'polyline',
            'points': cur_points,
            'closed': True,
            'fillColor': 'rgba(0,0,0,0)',
            'lineColor': 'rgb(0,255,0)'
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
    import psutil

    scheduler = getattr(args, 'scheduler', None)
    num_workers = getattr(args, 'num_workers', 0)
    num_threads_per_worker = getattr(args, 'num_threads_per_worker', 0)

    if scheduler == 'multithreading':
        from multiprocessing.pool import ThreadPool

        import dask.threaded

        if num_threads_per_worker <= 0:
            num_workers = max(1, psutil.cpu_count(logical=False) + num_threads_per_worker)
        else:
            num_workers = num_threads_per_worker
        print('Starting dask thread pool with %d thread(s)' % num_workers)
        dask.config.set(pool=ThreadPool(num_workers))
        dask.config.set(scheduler='threads')
        return

    if scheduler == 'multiprocessing':
        import multiprocessing

        import dask.multiprocessing

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


def get_region_polygons(region):
    """
    Convert the region into a list of polygons.

    Params
    ------
    region: list
        4 elements -- left, top, width, height -- or all -1, meaning the whole
        slide.
        6 elements -- x,y,z,rx,ry,rz: a rectangle specified by center and
        radius.
        8 or more elements -- x,y,x,y,...: a polygon.  The region is the
        bounding box.

    Returns
    -------
    polygons: list
        A list of lists of x, y tuples.
    """
    if len(region) % 2 or len(region) < 4:
        raise ValueError('region must be 4, 6, or a list of 2n values.')
    region = [float(v) for v in region]
    if region == [-1] * 4:
        return []
    if len(region) == 6:
        region = [region[0] - region[3], region[1] - region[4], region[3] * 2, region[4] * 2]
    if len(region) == 4:
        region = [
            region[0], region[1],
            region[0] + region[2], region[1],
            region[0] + region[2], region[1] + region[3],
            region[0], region[1] + region[3],
        ]
    polys = [[]]
    for idx, x in enumerate(region[::2]):
        y = region[idx * 2 + 1]
        if x == -1 and y == -1:
            polys.append([])
        elif not len(polys[-1]) or (x, y) != tuple(polys[-1][-1]):
            polys[-1].append((x, y))
    for poly in polys:
        if len(poly) and poly[0] == poly[-1]:
            poly[-1:] = []
    polys = [poly for poly in polys if len(poly) >= 3]
    return polys


def polygons_to_binary_mask(polygons, x=0, y=0, width=None, height=None):
    """Convert a set of region polygons to a numpy binary mask.

    Params
    ------
    polygons: list
        A list of lists of x, y tuples.  If None, returns None.
    x: integer
        An offset for the mask compared to the polygon coordinates.
    y: integer
        An offset for the mask compared to the polygon coordinates.
    width: integer
        The width of the mask to return.  None uses the maximum polygon
        coordinate.
    height: integer
        The height of the mask to return.  None uses the maximum polygon
        coordinate.

    Returns
    -------
    mask: numpy.array
        A 1-bit numpy array where 1 is inside an odd number of polygons.  This
        can return None if polygons was None.
    """
    import PIL.Image
    import PIL.ImageChops
    import PIL.ImageDraw

    if polygons is None or not len(polygons):
        return None
    if width is None:
        width = max(pt[0] for poly in polygons for pt in poly) - x
    if height is None:
        height = max(pt[1] for poly in polygons for pt in poly) - y
    full = PIL.Image.new('1', (width, height), 0)
    for polyidx, poly in enumerate(polygons):
        mask = PIL.Image.new('1', (width, height), 0)
        PIL.ImageDraw.Draw(mask).polygon(
            [(int(pt[0] - x), int(pt[1] - y)) for pt in poly], outline=None, fill=1)
        if not polyidx:
            full = mask
        else:
            full = PIL.ImageChops.logical_xor(full, mask)
    return np.array(full)


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
        6 elements -- x,y,z,rx,ry,rz: a rectangle specified by center and
        radius.
        8 or more elements -- x,y,x,y,...: a polygon.  The region is the
        bounding box.
    maxRegionSize: int, optional
        Maximum size permitted of any single dimension.  -1 or None for
        unlimited.
    tilesource: tilesource, optional
        A `large_image` tilesource (or anything with `.sizeX` and `.sizeY`
        properties) that is used to determine the size of the whole slide if
        necessary.  Must be provided if `maxRegionSize` is not None.

    Returns
    -------
    region_dict: dict
        Either {} (for the special region -1,-1,-1,-1) or
        {'region': region_subdict}

    """

    if len(region) % 2 or len(region) < 4:
        raise ValueError('region must be 4, 6, or a list of 2n values.')

    useWholeImage = region == [-1] * 4

    if len(region) == 6:
        region = [region[0] - region[3], region[1] - region[4], region[3] * 2, region[4] * 2]
    if len(region) >= 8:
        polygons = get_region_polygons(region)
        minx = min(pt[0] for poly in polygons for pt in poly)
        maxx = max(pt[0] for poly in polygons for pt in poly)
        miny = min(pt[1] for poly in polygons for pt in poly)
        maxy = max(pt[1] for poly in polygons for pt in poly)
        region = [minx, miny, maxx - minx, maxy - miny]

    if maxRegionSize is not None and maxRegionSize > 0:
        if tilesource is None:
            raise ValueError('tilesource must be provided if maxRegionSize is specified')
        if useWholeImage:
            size = max(tilesource.sizeX, tilesource.sizeY)
        else:
            size = max(region[-2:])
        if size > maxRegionSize:
            raise ValueError('Requested region is too large!  '
                             'Please see --maxRegionSize')
    # If a tilesource was specified, restrict the region to the image size
    if not useWholeImage and tilesource:
        minx = max(0, region[0])
        maxx = min(tilesource.sizeX, region[0] + region[2])
        miny = max(0, region[1])
        maxy = min(tilesource.sizeY, region[1] + region[3])
        region = [minx, miny, maxx - minx, maxy - miny]

    return {} if useWholeImage else dict(
        region=dict(zip(['left', 'top', 'width', 'height'],
                        [int(val) for val in region])))


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
