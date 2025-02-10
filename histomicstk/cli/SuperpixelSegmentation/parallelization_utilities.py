import sys
import time
from xml.sax.saxutils import escape
import large_image
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy
import pyvips
import multiprocessing.shared_memory
import functools
import operator
import itertools
import concurrent.futures

import math
import skimage
import scipy
import json
import large_image
import numpy as np
from PIL import Image
from typing import cast


class Mask:
    """A class for manipulating masks linked to whole-slide images.

    Given a low-resolution mask for a whole-slide image (WSI), this class
    provides precise cropping of corresponding WSI regions at any desired
    magnification.

    Parameters
    ----------
    file : str
        Path to the mask file. Must be a binary or grayscale image readable
        by PIL.
    wsi : large_image
        A large_image object.
    """

    def __init__(self, file, wsi):
        self.mask = Image.open(file)
        if self.mask.mode not in {"1", "L"}:
            raise ValueError(
                f"The mask ({mask_filename}) must be a grayscale or binary image."
            )
        self.wsi = wsi
        meta = wsi.getMetadata()
        if (
            abs(
                math.log(
                    (self.mask.size[0] / self.wsi.metadata["sizeX"])
                    / (self.mask.size[1] / self.wsi.metadata["sizeY"])
                )
            )
            > 0.20
        ):
            raise ValueError(
                "The mask aspect ratio does not match the whole slide image."
            )

        # create a cumulative mask for fast counting of mask coverage in rois
        self.cumulative = np.zeros(
            (self.mask.size[1] + 2, self.mask.size[0] + 2), dtype=np.int64
        )
        self.cumulative[1:-1, 1:-1] = np.array(self.mask) > 0
        self.cumulative = np.cumsum(np.cumsum(self.cumulative, axis=0), axis=1)

    def _scale(self, region, magnification):        
        # native scan magnification if level is none
        native = self.wsi.metadata["magnification"]
        magnification = native if magnification is None else magnification
                    
        # get size of wsi at requested magnification
        scale = cast(float, native) / cast(float, magnification)
        size_x = math.floor(self.wsi.sizeX / scale)
        size_y = math.floor(self.wsi.sizeY / scale)
        
        # crop mask to float bounding box of provided region
        scale_x = self.mask.size[0] / cast(float, size_x)
        scale_y = self.mask.size[1] / cast(float, size_y)

        # scaled mask region
        return (
            scale_x * region["left"],
            scale_y * region["top"],
            scale_x * (region["left"] + region["width"]),
            scale_y * (region["top"] + region["height"])
        )
            
    def getRegion(self, region, magnification=None, format='numpy'):
        """Read a region from mask corresponding to a magnification/level and region
        of the large_image object.

        Parameters
        ----------
        region : dict
            A dictionary with integer values 'left', 'top', 'width', and 'height' 
            interpreted as pixels at the provided magnification or level.
        magnification : float
            The magnification to return the mask region at. Default value of None
            indicates that native scan magnification will be used.
        format : {'numpy', 'pil'}
            String indicating whether to return a numpy ndarray (default) or a 
            PIL.Image.
            
        Returns
        -------
        tile : PIL.Image or numpy.ndarray
            The cropped and scaled mask corresponding to `region` at `magnification`.
        """

        # mask region (float) corresponding to requested wsi region
        mask_region = self._scale(region, magnification)
        
        # crop mask to float bounding box of provided region - 
        cropped = self.mask.crop(mask_region).resize((region["width"], region["height"]))
        return np.array(cropped) if format == 'numpy' else cropped

    def maskedFraction(self, region, magnification=None):
        """Determine the fraction of positive corresponding mask pixels given a region and 
        magnification/level of a large_image object.

        Parameters
        ----------
        region : dict
            A dictionary with integer values 'left', 'top', 'width', and 'height' 
            interpreted as pixels at the provided magnification or level.
        magnification : float
            The magnification to return the mask region at. Default value of None
            indicates that native scan magnification will be used.
            
        Returns
        -------
        empty : float
            The fraction of the region occupied by mask pixels > 0.
        """
        def aceil(x):
            return math.floor(x+1)
    
        def linear(x, slice):
            x1 = math.floor(cast(float,x))
            x2 = aceil(cast(float,x))
            den = (x2-x1)
            return (x2-cast(float,x)) * slice[x1] / den + (cast(float,x)-x1) * slice[x2] / den
        
        def bilinear(x, y, cumulative):
            y1 = math.floor(cast(float,y))
            y2 = aceil(cast(float,y))
            f1 = linear(x, cumulative[y1,:])
            f2 = linear(x, cumulative[y2,:])
            den = (y2-y1)
            return (y2-cast(float,y)) * f1 / den + (cast(float,y)-y1) * f2 / den

        # mask region (float) corresponding to requested wsi region
        mr = self._scale(region, magnification)

        # measure positivity in region
        total = bilinear(mr[0], mr[1], self.cumulative) + \
        bilinear(mr[2], mr[3], self.cumulative) - \
        bilinear(mr[0], mr[3], self.cumulative) - \
        bilinear(mr[2], mr[1], self.cumulative)
        return total / ((mr[2]-mr[0]) * (mr[3]-mr[1]))


class SharedNumpyArray:
    def __init__(self, shape, dtype):
        """Init"""
        self.shape = shape
        self.dtype = numpy.dtype(dtype)
        self.shm_size = functools.reduce(operator.mul, shape, 1) * self.dtype.itemsize
        self.shm = multiprocessing.shared_memory.SharedMemory(
            create=True, size=self.shm_size)
        self.created = True

    def copy(self, arr):
        self.shape = arr.shape
        self.buf = numpy.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
        self.buf[:] = arr[:]
    
    def tobytes(self):
        return self.buf.tobytes()

    def pack(self, arrays):
        """Pack arrays along 0-axis"""
        self.shape = (len(arrays), *arrays[0].shape)
        self.buf = numpy.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
        for i, arr in enumerate(arrays):
            self.buf[i] = arr

    def unpack(self):
        """Return a view of 0-axis slices"""
        slice_shape = list(self.shape)[1:]
        slice_bytes = functools.reduce(operator.mul, slice_shape, 1) * \
            self.dtype.itemsize
        slices = [
            numpy.ndarray(
                slice_shape, 
                dtype=self.dtype,
                buffer=self.shm.buf, 
                offset=i*slice_bytes
            )
            for i in range(0, self.shape[0])
        ]
        return slices

    # If we want easier interoperability, we could, instead, forward a
    # whitelist of attributes to our underlying np.ndarray object; these could
    # be enumerated and done via __getattribute__
    def __getitem__(self, idx):
        return self.buf[idx]

    def __array__(self, dtype=None):
        return self.buf.copy().astype(dtype) if dtype is not None else self.buf.copy()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['shm']
        state.pop('buf', None)
        state['created'] = False
        state['shm_name'] = self.shm.name
        return state

    def __setstate__(self, state):
        state = state.copy()
        shm_name = state.pop('shm_name')
        self.__dict__.update(state)
        self.shm = multiprocessing.shared_memory.SharedMemory(shm_name)
        self.buf = numpy.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)

    def __del__(self):
        if hasattr(self, 'shm'):
            self.shm.close()
            if getattr(self, 'created', None) is True:
                self.shm.unlink()



class ProgressHelper:
    def __init__(self, name, comment='', report=True):
        """
        A name is required.  The comment is shown in the popup as commentary
        on the task.
        """
        self._name = name
        self._comment = comment
        self._start = time.time()
        self._last = 0
        self._report = report

    def __enter__(self):
        if self._report:
            print("""<filter-start>
                <filter-name>%s</filter-name>
                <filter-comment>%s</filter-comment>
                </filter-start>""" % (escape(self._name), escape(self._comment)))
            sys.stdout.flush()
        self._start = time.time()
        return self

    def items(self, items):
        self._items = items
        self._item_progress = []

    def item_progress(self, item, val):
        if item not in self._items:
            return
        idx = self._items.index(item)
        if len(self._item_progress) < len(self._items):
            self._item_progress += [0] * (len(self._items) - len(self._item_progress))
        self._item_progress[idx] = val
        total = sum(self._item_progress) / len(self._item_progress)
        self.progress(total)

    def progress(self, val):
        if self._report and (val == 0 or val == 1 or time.time() - self._last >= 0.1):
            print("""<filter-progress>%s</filter-progress>""" % val)
            sys.stdout.flush()
            self._last = time.time()

    def message(self, comment):
        self._comment = comment
        if self._report:
            print("""<filter-comment>%s</filter-comment>""" % escape(comment))
            sys.stdout.flush()

    def name(self, name):
        # Leave the original name alone
        if self._report:
            print("""<filter-name>%s</filter-name>""" % escape(name))
            sys.stdout.flush()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        duration = end - self._start
        if self._report:
            print("""<filter-end>
                <filter-name>%s</filter-name>
                <filter-time>%s</filter-time>
                </filter-end>""" % (escape(self._name), duration))
            sys.stdout.flush()


__all__ = ['ProgressHelper']




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
        msg = 'region must be 4, 6, or a list of 2n values.'
        raise ValueError(msg)
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
        msg = 'region must be 4, 6, or a list of 2n values.'
        raise ValueError(msg)

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
            msg = 'tilesource must be provided if maxRegionSize is specified'
            raise ValueError(msg)
        if useWholeImage:
            size = max(tilesource.sizeX, tilesource.sizeY)
        else:
            size = max(region[-2:])
        if size > maxRegionSize:
            msg = (
                'Requested region is too large!  '
                'Please see --maxRegionSize'
            )
            raise ValueError(msg)
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




def first_order(tasks):
    """ task first-order dependents and ancestors """
    
    dependents = {}
    ancestors = {}
    
    # get range of tasks
    xmax = max([t[0] for t in tasks])
    ymax = max([t[1] for t in tasks])
    
    # # four neighborhood dependencies
    # anc_mapping = {
    #     (0, 0): [],
    #     (0, 1): [(0, -1), (0, 1)],
    #     (1, 0): [(-1, 0), (1, 0)],
    #     (1, 1): [(-1, 0), (0, -1), (1, 0), (0, 1)],
    # }
    # dep_mapping = {
    #     (0, 0): [(-1, 0), (0, -1), (1, 0), (0, 1)],
    #     (0, 1): [(1, 0), (-1, 0)],
    #     (1, 0): [(0, 1), (0, -1)],
    #     (1, 1): [],
    # }

    # eight neighbourhood dependencies
    anc_mapping = {
        (0, 0): [],
        (0, 1): [(0, -1), (0, 1)],
        (1, 0): [(-1, 0), (1, 0), (1,1),(1,-1),(-1,1),(-1,-1)],
        (1, 1): [(-1, 0), (0, -1), (1, 0), (0, 1), (1, 1), (-1, -1), (1, -1), (-1, 1)],
    }
    dep_mapping = {
        (0, 0): [(-1, 0), (0, -1), (1, 0), (0, 1), (1, 1), (-1, -1), (1, -1), (-1, 1)],
        (0, 1): [(1, 0), (-1, 0), (1,1), (1, -1), (-1, 1), (-1,-1)],
        (1, 0): [(0, 1), (0, -1)],
        (1, 1): [],
    }
    
    
    # handle borders - keep dependencies in the range of possible tasks (>=0 and <= max)
    for task in tasks:
        deps = dep_mapping[(task[0]%2, task[1]%2)]
        dependents[task] = [
            (task[0]+d[0], task[1]+d[1]) for d in deps
            if (task[0]+d[0], task[1]+d[1]) in tasks
        ]
        ancs = anc_mapping[(task[0]%2, task[1]%2)]
        ancestors[task] = [
            (task[0]+a[0], task[1]+a[1]) for a in ancs
            if (task[0]+a[0], task[1]+a[1]) in tasks
        ]
    return ancestors, dependents


def tile_grid(ts, opts, averageSize, overlap, tileSize, tiparams, verbose=False):
    """ returns a dense 2D grid of tasks over (h,w) grid """
    scale = 1
    if opts.magnification:
        tiparams['scale'] = {'magnification': opts.magnification}


    meta = ts.getMetadata()
    task_ids = []
    tasks = {}

    ii = 1

    for tile in ts.tileIterator(
        format=large_image.constants.TILE_FORMAT_NUMPY,
        tile_size=dict(width=tileSize, height=tileSize),
        tile_overlap=dict(x=overlap, y=overlap),
        **tiparams,
    ):
        if verbose:
            print(f'Fetching TILE # {ii}') 

        if meta['magnification'] and tile['magnification']:
            scale = meta['magnification'] / tile['magnification']
        x0 = tiparams.get('region', {}).get('left', 0)
        y0 = tiparams.get('region', {}).get('top', 0)


        tx0 = int((tile['gx'] - x0) / scale)
        ty0 = int((tile['gy'] - y0) / scale)

        if verbose:
            print((tx0, ty0))
        task_ids.append((tx0, ty0))
        tasks[(tx0, ty0)] = (x0, y0, scale, tile)
        ii += 1

    x_s = set([idx[0] for idx in task_ids])
    y_s = set([idx[1] for idx in task_ids])

    h, w = len(y_s),len(x_s)

    coordx = {}
    coordy = {}

    i,j = 0,0
    for x,y in task_ids:
        if x not in coordx.values():
            coordx[i] = x
            i+=1
        if y not in coordy.values():
            coordy[j] = y
            j+=1

    return (task_ids, tasks, 
            h, w, [(x, y) for x, y in itertools.product(range(w), range(h))], 
            coordx, coordy)


def tile_grid_w_mask(ts, mask, opts, averageSize, overlap, tileSize, tiparams, verbose=False):
    """ returns a dense 2D grid of tasks over (h,w) grid """
    scale = 1
    if opts.magnification:
        tiparams['scale'] = {'magnification': opts.magnification}


    meta = ts.getMetadata()
    all_tile_ids = []
    alltile_metadata = {}
    task_ids = []
    tasks = {}

    ii = 1
    tile_shape_times = []
    mask_gen_times = []

    start = time.time()
    for tile in ts.tileIterator(
        format=large_image.constants.TILE_FORMAT_NUMPY,
        tile_size=dict(width=tileSize, height=tileSize),
        tile_overlap=dict(x=overlap, y=overlap),
        **tiparams,
    ):
        if verbose:
            print(f'Fetching TILE # {ii}') 

        if meta['magnification'] and tile['magnification']:
            scale = meta['magnification'] / tile['magnification']
        x0 = tiparams.get('region', {}).get('left', 0)
        y0 = tiparams.get('region', {}).get('top', 0)


        tx0 = int((tile['gx'] - x0) / scale)
        ty0 = int((tile['gy'] - y0) / scale)

        # print(tile['tile'].shape, (tile['height'], tile['width']))

        start_ts = time.time()
        height, width, _ = tile['tile'].shape
        stop_ts = time.time()
        tile_shape_times.append(stop_ts-start_ts)

        magnification = tile['magnification']
        region=dict(
            left=tx0, 
            top=ty0, 
            width=width, height=height
        )


        alltile_metadata[(tx0, ty0)] = (height, width)
                
        start_mc = time.time()
        mask_crop = mask.getRegion(region, magnification, 'pil')

        # calculate fraction of in-mask pixels in this region
        masked_fraction = mask.maskedFraction(region, magnification)

        mask_crop = numpy.array(mask_crop)>0
        stop_mc = time.time()
        mask_gen_times.append(stop_mc-start_mc)

        # fig, axes = plt.subplots(1, 2)
        # print(ii,':',numpy.array(mask_crop).sum())
        # axes[0].imshow(tile['tile'], cmap='viridis')
        # axes[1].imshow(mask_crop, cmap='viridis')
        # plt.show()

        all_tile_ids.append((tx0, ty0))



        if masked_fraction > 0.1:
            if verbose:
                print((tx0, ty0))
            task_ids.append((tx0, ty0))
            
            tasks[(tx0, ty0)] = (x0, y0, scale, tile, mask_crop)
            
        else:
            if verbose:
                print(f'Dropping tile at {(tx0, ty0)}. No tissue found.')
        
        ii += 1
    stop = time.time()

    print('time elasped mask:', stop - start)

    print('tile shape time:', sum(tile_shape_times) )

    print('mask crop time:', sum(mask_gen_times) )

    x_s = set([idx[0] for idx in all_tile_ids])
    y_s = set([idx[1] for idx in all_tile_ids])

    h, w = len(y_s),len(x_s)

    coordx = {}
    coordy = {}

    i,j = 0,0
    for x,y in all_tile_ids:
        if x not in coordx.values():
            coordx[i] = x
            i+=1
        if y not in coordy.values():
            coordy[j] = y
            j+=1

        

    return (task_ids, tasks, 
            h, w, 
            [(x, y) for x, y in itertools.product(range(w), range(h)) 
             if (coordx[x], coordy[y]) in task_ids], scale,
            coordx, coordy,
            alltile_metadata)


def get_trim_dict(dependents):
    trim_dict = {d:[] for d in dependents.keys()}
    for d in dependents.keys():
        for i in dependents[d]:
            vec = (i[0]-d[0],i[1]-d[1])
            if abs(vec[0])+abs(vec[1]) == 1:
                if vec[0] == 1:
                    trim_dict[d].append('right')
                elif vec[0] == -1:
                    trim_dict[d].append('left')
                elif vec[1] == 1:
                    trim_dict[d].append('bottom')
                elif vec[1] == -1:
                    trim_dict[d].append('top')
    return trim_dict


def get_ancestor_tileids(ancestors, coordx, coordy):
    ancestor_taskids = {}

    for k in ancestors.keys():
        # print(k,':',ancestors[k])
        ancestor_taskids[(coordx[k[0]], coordy[k[1]])] = [(coordx[ids[0]], coordy[ids[1]]) for ids in ancestors[k]]

    return ancestor_taskids



def plot_tasks(tasks, data, arrows=None, coordx=None,coordy=None,zoom=0.01):
    """ display task grid - optionally show dependencies"""
    kwargs = {"x": [t[0] for t in tasks], "y": [t[1] for t in tasks]}
    # Add images
    fig, ax = plt.subplots()
    for x0, y0 in tasks:
        img = data[(coordx[x0], coordy[y0])][3]['tile']
        im = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        ax.add_artist(ab)
        
    if arrows:
        for t in tasks:
            for a in arrows[t]:
                ax.arrow(
                    t[0], t[1], 0.9*(a[0]-t[0]), 0.9*(a[1]-t[1]), 
                    length_includes_head=True, width=0.005,
                    head_width=0.1,
                )
    plt.gca().invert_yaxis()
    plt.axis('equal')

def np_to_vips(strip):
        # print(f'converting numpy to pyvips | task id: {(coordx[task_id[0]], coordy[task_id[1]])} | keys: {k}')
    data = strip
    vimg = pyvips.Image.new_from_memory(
    numpy.ascontiguousarray(data).data,
    data.shape[1], data.shape[0], data.shape[2],
    large_image.constants.dtypeToGValue[data.dtype.char])
    vimg = vimg.copy(interpretation=pyvips.Interpretation.RGB)
    vimgTemp = pyvips.Image.new_temp_file('%s.v')
    vimg.write(vimgTemp)
    vimg = vimgTemp

    return vimg

def compute_mask(task_id, task, overlap, strips_local, trim_dict, coordx, coordy, ancestor_taskids):

    #convert numpy to pyvips

    task_id = (coordx[task_id[0]], coordy[task_id[1]])
    (tx0, ty0) = task_id
    (x0, y0, scale, tile) = task#[task_id]

    ancestors = ancestor_taskids[task_id]

    # print('tile:', tx0, ty0, ', ancestors:', ancestors, ', trim_edge:', trims)

    img = tile['tile']
    n_pixels = tile['width'] * tile['height']
    mask = None


    if overlap:
        mask = numpy.ones(img.shape[:2])
        for k in strips_local.keys():

            y, simg = strips_local[k]

            # print(k)

            if (k[0],y) in ancestors:


                try:
                    simg = np_to_vips(simg)
                    subx = max(0, k[0] - tx0)  
                    suby = max(0, y - ty0)
                    simg_h = simg.height
                    simg_w = simg.width
                    if suby:
                        crop_height = tile['height'] - abs(ty0 - y)
                    else:
                        crop_height = min(tile['height'], simg_h - abs(ty0 - y))

                    if subx:
                        crop_width = tile['width'] - abs(tx0 - k[0])
                    else:
                        crop_width = min(tile['width'], simg_w - abs(tx0 - k[0]))

                    subimg = simg.crop(
                        max(0, tx0-k[0]),
                        max(0, ty0 - y),
                        crop_width,
                        crop_height)
                    # Our mask is true when a pixel has not been set
                    submask = numpy.ndarray(
                        buffer=subimg[3].write_to_memory(),
                        dtype=numpy.uint8,
                        shape=[subimg.height, subimg.width]) == 0

                    mask[suby:suby + submask.shape[0], subx:subx + submask.shape[1]] *= submask

                    # print(f'{simg}\n, mask shape:{img.shape}, tile:{tx0},{ty0}, strip(x,y):{(k[0],y)},subx:{subx}-{simg_w} {abs(tx0 - k[0])},suby:{suby}-{simg_h} {abs(ty0 - y)},{crop_width},{crop_height}')

                except Exception as e:
                    print(e)
                    print(f'{simg}\n, mask shape:{img.shape}, tile:{tx0},{ty0}, strip(x,y):{(k[0],y)},subx:{subx}-{simg_w} {abs(tx0 - k[0])},suby:{suby}-{simg_h} {abs(ty0 - y)},{crop_width},{crop_height}')
                    raise ValueError


    return mask


def compute_mask_novips(task_id, task, overlap, strips_local, coordx, coordy, ancestor_taskids):


    task_id = (coordx[task_id[0]], coordy[task_id[1]])
    (tx0, ty0) = task_id

    if len(task)==5:
        (x0, y0, scale, tile, tile_mask) = task
    else:
        (x0, y0, scale, tile) = task


    ancestors = ancestor_taskids[task_id]


    img = tile['tile']
    n_pixels = tile['width'] * tile['height']
    mask = None


    if overlap:
        mask = numpy.ones(img.shape[:2])

        mask = mask*(numpy.array(tile_mask)>0)

        for k in strips_local.keys():

            y, simg = strips_local[k]

            if (k[0],y) in ancestors:


                try:
                    subx = max(0, k[0] - tx0)  
                    suby = max(0, y - ty0)
                    simg_h, simg_w = simg.shape[0], simg.shape[1]
                    if suby:
                        crop_height = tile['height'] - abs(ty0 - y)
                    else:
                        crop_height = min(tile['height'], simg_h - abs(ty0 - y))

                    if subx:
                        crop_width = tile['width'] - abs(tx0 - k[0])
                    else:
                        crop_width = min(tile['width'], simg_w - abs(tx0 - k[0]))

                    subimg = simg[
                        max(0, ty0 - y):max(0, ty0 - y)+crop_height,
                        max(0, tx0-k[0]):max(0, tx0-k[0])+crop_width, 1
                        ]
                    # # Our mask is true when a pixel has not been set
                    submask = subimg == 0

                    mask[suby:suby + submask.shape[0], subx:subx + submask.shape[1]] *= submask

                except Exception as e:
                    print(e)
                    print(f'{simg}\n, mask shape:{img.shape}, tile:{tx0},{ty0}, strip(x,y):{(k[0],y)},subx:{subx}-{simg_w} {abs(tx0 - k[0])},suby:{suby}-{simg_h} {abs(ty0 - y)},{crop_width},{crop_height}')
                    raise ValueError


    return mask


def process_tiles(task_ids, tasks, grid, coordx, coordy, trim_dict, ancestors, ancestor_taskids, dependents, opts, overlap, num_workers):
    
    results = {}
    strips = {}
    strips_found = {}
    bboxes_dict = {}
    bboxesUser_dict = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as pool:
    
    # submit all tasks with no ancestors 

        submitted = {
            pool.submit(tilejob, task, tasks[(coordx[task[0]], coordy[task[1]])], trim_dict, coordx, coordy, 
                        numpy.array(tasks[(coordx[task[0]], coordy[task[1]])][4]), opts): 
                        task for task in [t for t in grid if not len(ancestors[t])]
        }
        
        # as tasks complete, update dependencies and submit new tasks
        for i in range(len(grid)):

            completed, waiting = concurrent.futures.wait(       
                submitted, return_when=concurrent.futures.FIRST_COMPLETED
            )

            for future in completed:
                results[submitted[future]] = future.result() 

                strips[results[submitted[future]][0]] = results[submitted[future]][1]
                strips_found[results[submitted[future]][0]] = results[submitted[future]][2]
                bboxes_dict[results[submitted[future]][0]] = results[submitted[future]][3]
                bboxesUser_dict[results[submitted[future]][0]] = results[submitted[future]][4]

                for dependent in dependents[submitted[future]]:
                    try:
                        ancestors[dependent].remove(submitted[future])
                    except Exception as e:
                        print(e)
                        print(dependent,'\n', submitted[future])
                        raise Exception

                    if not len(ancestors[dependent]):
                        mask = compute_mask_novips(dependent, tasks[(coordx[dependent[0]], coordy[dependent[1]])], overlap, strips, coordx, coordy, ancestor_taskids)

                        submitted[pool.submit(tilejob, dependent, tasks[(coordx[dependent[0]], coordy[dependent[1]])], trim_dict, coordx, coordy, mask, opts)] = dependent

                del submitted[future]
    
    return strips, strips_found, bboxes_dict, bboxesUser_dict


def tilejob(task_id, task, trim_dict, coordx, coordy, mask, opts):

    averageSize = opts.superpixelSize ** 2
    overlap = opts.superpixelSize * 4 * 2 if opts.overlap else 0 
    tileSize = opts.tileSize + overlap


    found = 0
    bboxes = []
    bboxesUser = []
    

    trims = trim_dict[task_id]

    task_id = (coordx[task_id[0]], coordy[task_id[1]])
    (tx0, ty0) = task_id

    # print(task_id)

    if len(task) == 5:
        (x0, y0, scale, tile, tile_mask) = task
    else:
        (x0, y0, scale, tile) = task


    img = tile['tile']
    n_pixels = tile['width'] * tile['height']

    if mask is None:
        mask = numpy.ones(img.shape[:2])


    if overlap:
        n_pixels = numpy.count_nonzero(mask)



    n_segments = math.ceil(n_pixels / averageSize)

    # print(n_segments)

    segments = skimage.segmentation.slic(
        img,
        n_segments=n_segments,
        slic_zero=bool(opts.slic_zero),
        compactness=opts.compactness,
        sigma=opts.sigma,
        start_label=0,
        enforce_connectivity=True,
        mask=mask,
    )

    # if numpy.any(segments <= 0):
    # print('immediate:', segments.min(), segments.max(), numpy.any(segments == 0))

    # We now have an array that is the same size as the image
    maxValue = numpy.max(segments) + 1
 
    if overlap:
        # Keep any segment that is at all in the non-overlap region

        if 'right' in trims:
            ridx = tile['width'] - tile['tile_overlap']['right']
        else:
            ridx = -1
            
        if 'bottom' in trims:
            bidx = tile['height'] - tile['tile_overlap']['bottom']
        else:
            bidx = -1

        
        if 'left' in trims:    
            lidx = tile['tile_overlap']['left']
        else:
            lidx = 0


        if 'top' in trims:    
            tidx = tile['tile_overlap']['top']
        else:
            tidx = 0
            
        
        core = segments[
            tidx:bidx,
            lidx:ridx]
        
        coremask = mask[
            tidx:bidx,
            lidx:ridx]


        core[numpy.where(coremask != 1)] = -1
        usedIndices = numpy.unique(core)
        usedIndices = numpy.delete(usedIndices, numpy.where(usedIndices < 0))
        usedLut = [-1] * maxValue
        for idx, used in enumerate(usedIndices):
            if used >= 0:
                usedLut[used] = idx
        usedLut = numpy.array(usedLut, dtype=int)

     
        maxValue = len(usedIndices)
        segments = usedLut[segments]
        mask *= (segments != -1)

        # if numpy.any(segments < 0):
        #     print('after core:', segments.min(), segments.max(), numpy.any(segments == 0))

    if str(opts.bounding).lower() not in {'', 'none'}:
        regions = skimage.measure.regionprops(1 + segments)
        for _pidx, props in enumerate(regions):
            by0, bx0, by1, bx1 = props.bbox
            bboxes.append((
                ((bx0 + bx1) / 2 + tx0) * scale + x0,
                ((by0 + by1) / 2 + ty0) * scale + y0,
                (bx1 - bx0) * scale,
                (by1 - by0) * scale))
            bboxesUser.extend([
                (bx0 + tx0) * scale + x0,
                (by0 + ty0) * scale + y0,
                (bx1 + tx0) * scale + x0,
                (by1 + ty0) * scale + y0,
            ])

    if opts.boundaries:
        segments *= 2
        maxValue *= 2
        edges = (scipy.ndimage.sobel(segments, axis=0) != 0) | (
            scipy.ndimage.sobel(segments, axis=1) != 0)
        edges[0, :] = True
        edges[-1, :] = True
        edges[:, 0] = True
        edges[:, -1] = True
        segments += edges

    segments += found

    # if found > 0:
    #     print('found')

    # if numpy.any(segments < 0):
    #     print('after bound:', segments.min(), segments.max(), numpy.any(segments == 0))

    found += int(maxValue)
    if mask is None:
        data = numpy.dstack((
            (segments).astype(int))).astype('B')
        #numpy.where(segments >=0, segments+1, segments)*tile_mask

    else:
        data = numpy.dstack((
            (segments).astype(int),
            mask * 255)).astype('B')
        #numpy.where(segments >=0, segments+1, segments)*tile_mask

    # print(data[:,:,0].min(), data[:,:,1].max()) 

    x = tx0
    ty = tile['tile_position']['region_y']


    if hasattr(opts, 'callback'):
        opts.callback('tiles', tile['tile_position']['position'] + 1,
                        tile['iterator_range']['position'])
    
    return (x,ty), [ty0, data], found, bboxes, bboxesUser

