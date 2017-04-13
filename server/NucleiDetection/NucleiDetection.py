import histomicstk.preprocessing.color_normalization as htk_cnorm
import histomicstk.preprocessing.color_deconvolution as htk_cdeconv
import histomicstk.filters.shape as htk_shape_filters
import histomicstk.segmentation as htk_seg
import histomicstk.utils as htk_utils

import large_image

import dask
from dask.distributed import Client, LocalCluster
import multiprocessing

import os
import numpy as np
import json
import scipy as sp
import skimage.io
import skimage.measure
import itertools
import time

from ctk_cli import CLIArgumentParser

import logging
logging.basicConfig()

def detect_nuclei_kofahi(im_input, args):

    # perform color normalization
    im_nmzd = htk_cnorm.reinhard(im_input,
                                 args.reference_mu_lab,
                                 args.reference_std_lab)

    # perform color decovolution
    w = htk_cdeconv.utils.get_stain_matrix(args)

    im_stains = htk_cdeconv.color_deconvolution(im_nmzd, w).Stains

    im_nuclei_stain = im_stains[:, :, 0].astype(np.float)

    # segment foreground (assumes nuclei are darker on a bright background)
    im_nuclei_fgnd_mask = sp.ndimage.morphology.binary_fill_holes(
        im_nuclei_stain < args.foreground_threshold)

    # run adaptive multi-scale LoG filter
    im_log = htk_shape_filters.clog(im_nuclei_stain, im_nuclei_fgnd_mask,
                                    sigma_min=args.min_radius / np.sqrt(2),
                                    sigma_max=args.max_radius / np.sqrt(2))

    # apply local maximum clustering
    im_nuclei_seg_mask, seeds, max = htk_seg.nuclear.max_clustering(
        im_log, im_nuclei_fgnd_mask, args.local_max_search_radius)

    # filter out small objects
    im_nuclei_seg_mask = htk_seg.label.area_open(
        im_nuclei_seg_mask, args.min_nucleus_area).astype(np.int)

    return im_nuclei_seg_mask


def detect_tile_nuclei(slide_path, tile_position, args, **it_kwargs):

    # get slide tile source
    ts = large_image.getTileSource(slide_path)

    # get requested tile
    tile_info = ts.getSingleTile(tile_position=tile_position,
                                 **it_kwargs)

    # get tile image
    im_tile = tile_info['tile'][:, :, :3]

    # segment nuclei
    im_nuclei_seg_mask = detect_nuclei_kofahi(im_tile, args)

    # generate nuclei bounding boxes annotations
    obj_props = skimage.measure.regionprops(im_nuclei_seg_mask)

    nuclei_bbox_list = []

    gx = tile_info['gx']
    gy = tile_info['gy']
    wfrac = tile_info['gwidth'] / np.double(tile_info['width'])
    hfrac = tile_info['gheight'] / np.double(tile_info['height'])

    for i in range(len(obj_props)):

        cx = obj_props[i].centroid[1]
        cy = obj_props[i].centroid[0]
        width = obj_props[i].bbox[3] - obj_props[i].bbox[1] + 1
        height = obj_props[i].bbox[2] - obj_props[i].bbox[0] + 1

        # convert to base pixel coords
        cx = np.round(gx + cx * wfrac, 2)
        cy = np.round(gy + cy * hfrac, 2)
        width = np.round(width * wfrac, 2)
        height = np.round(height * hfrac, 2)

        # create annotation json
        cur_bbox = {
            "type":        "rectangle",
            "center":      [cx, cy, 0],
            "width":       width,
            "height":      height,
            "rotation":    0,
            "fillColor":   "rgba(0,0,0,0)"
        }

        nuclei_bbox_list.append(cur_bbox)

    return nuclei_bbox_list


def compute_tile_foreground_fraction(slide_path, tile_position,
                                     im_fgnd_mask_lres, fgnd_seg_scale,
                                     **it_kwargs):

    # get slide tile source
    ts = large_image.getTileSource(slide_path)

    # get requested tile
    tile = ts.getSingleTile(tile_position=tile_position,
                            **it_kwargs)

    # get current region in base_pixels
    rgn_hres = {'left': tile['gx'], 'top': tile['gy'],
                'right': tile['gx'] + tile['gwidth'],
                'bottom': tile['gy'] + tile['gheight'],
                'units': 'base_pixels'}

    # get foreground mask for current tile at low resolution
    rgn_lres = ts.convertRegionScale(rgn_hres,
                                     targetScale=fgnd_seg_scale,
                                     targetUnits='mag_pixels')

    top = np.int(rgn_lres['top'])
    bottom = np.int(rgn_lres['bottom'])
    left = np.int(rgn_lres['left'])
    right = np.int(rgn_lres['right'])

    im_tile_fgnd_mask_lres = im_fgnd_mask_lres[top:bottom, left:right]

    # compute foreground fraction
    cur_fgnd_frac = im_tile_fgnd_mask_lres.mean()

    if np.isnan(cur_fgnd_frac):
        cur_fgnd_frac = 0

    return cur_fgnd_frac


def collect(x):
    return x


def disp_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def main(args):

    total_start_time = time.time()

    print('\n>> CLI Parameters ...\n')

    print args

    if not os.path.isfile(args.inputImageFile):
        raise IOError('Input image file does not exist.')

    if len(args.reference_mu_lab) != 3:
        raise ValueError('Reference Mean LAB should be a 3 element vector.')

    if len(args.reference_std_lab) != 3:
        raise ValueError('Reference Stddev LAB should be a 3 element vector.')

    if len(args.analysis_roi) != 4:
        raise ValueError('Analysis ROI must be a vector of 4 elements.')

    if np.all(np.array(args.analysis_roi) == -1):
        process_whole_image = True
    else:
        process_whole_image = False

    #
    # Initiate Dask client
    #
    print('\n>> Creating Dask client ...\n')

    scheduler_address = json.loads(args.scheduler_address)

    if scheduler_address is None:

        scheduler_address = LocalCluster(
            n_workers=multiprocessing.cpu_count()-1,
            scheduler_port=0,
            silence_logs=False
        )

    c = Client(scheduler_address)
    print c

    #
    # Read Input Image
    #
    print('\n>> Reading input image ... \n')

    ts = large_image.getTileSource(args.inputImageFile)

    ts_metadata = ts.getMetadata()

    print json.dumps(ts_metadata, indent=2)

    is_wsi = ts_metadata['magnification'] is not None

    #
    # Compute tissue/foreground mask at low-res for whole slide images
    #
    if is_wsi:

        print('\n>> Computing tissue/foreground mask at low-res ...\n')

        # get image at low-res
        maxSize = max(ts_metadata['sizeX'], ts_metadata['sizeY'])

        downsample_factor = 2**np.floor(np.log2(maxSize / 2048))

        fgnd_seg_mag = ts_metadata['magnification'] / downsample_factor

        fgnd_seg_scale = {'magnification': fgnd_seg_mag}

        im_lres, _ = ts.getRegion(
            scale=fgnd_seg_scale,
            format=large_image.tilesource.TILE_FORMAT_NUMPY
        )

        im_lres = im_lres[:, :, :3]

        # compute foreground mask at low-res
        im_fgnd_mask_lres = htk_utils.simple_mask(im_lres)

    #
    # Compute foreground fraction of tiles in parallel using Dask
    #
    tile_fgnd_frac_list = [1.0]

    it_kwargs = {
        'format': large_image.tilesource.TILE_FORMAT_NUMPY,
        'tile_size': {'width': args.analysis_tile_size},
        'scale': {'magnification': args.analysis_mag},
    }

    if not process_whole_image:

        it_kwargs['region'] = {
            'left':   args.analysis_roi[0],
            'top':    args.analysis_roi[1],
            'width':  args.analysis_roi[2],
            'height': args.analysis_roi[3],
            'units':  'base_pixels'
        }

    if is_wsi:

        print('\n>> Computing foreground fraction of all tiles ...\n')

        start_time = time.time()

        num_tiles = ts.getSingleTile(**it_kwargs)['iterator_range']['position']

        print 'Number of tiles = %d' % num_tiles

        tile_fgnd_frac_list = [None] * num_tiles

        for tile_position in range(num_tiles):

            tile_fgnd_frac_list[tile_position] = dask.delayed(compute_tile_foreground_fraction)(
                args.inputImageFile, tile_position,
                im_fgnd_mask_lres,
                fgnd_seg_scale,
                **it_kwargs
            )

        tile_fgnd_frac_cgraph = dask.delayed(collect)(tile_fgnd_frac_list)
        tile_fgnd_frac_list = np.array(tile_fgnd_frac_cgraph.compute())

        num_fgnd_tiles = np.count_nonzero(
            tile_fgnd_frac_list >= args.min_fgnd_frac)

        percent_fgnd_tiles = 100.0 * num_fgnd_tiles / num_tiles

        fgnd_frac_comp_time = time.time() - start_time

        print 'Number of foreground tiles = %d (%.2f%%)' % (
            num_fgnd_tiles, percent_fgnd_tiles)

        print 'Time taken = %s' % disp_time(fgnd_frac_comp_time)

    #
    # Detect nuclei in parallel using Dask
    #
    print('\n>> Detecting nuclei ...\n')

    start_time = time.time()

    tile_nuclei_list = []

    for tile in ts.tileIterator(**it_kwargs):

        tile_position = tile['tile_position']['position']

        if is_wsi and tile_fgnd_frac_list[tile_position] <= args.min_fgnd_frac:
            continue

        # detect nuclei
        cur_nuclei_list = dask.delayed(detect_tile_nuclei)(
            args.inputImageFile,
            tile_position,
            args, **it_kwargs)

        # append result to list
        tile_nuclei_list.append(cur_nuclei_list)

    nuclei_detection_time = time.time() - start_time

    tile_nuclei_list = dask.delayed(collect)(tile_nuclei_list).compute()

    nuclei_list = list(itertools.chain.from_iterable(tile_nuclei_list))

    print 'Number of nuclei = ', len(nuclei_list)
    print "Time taken = %s" % disp_time(nuclei_detection_time)

    #
    # Write annotation file
    #
    print('\n>> Writing annotation file ...\n')

    annot_fname = os.path.splitext(
        os.path.basename(args.outputNucleiAnnotationFile))[0]

    annotation = {
        "name":     annot_fname,
        "elements": nuclei_list
    }

    with open(args.outputNucleiAnnotationFile, 'w') as annotation_file:
        json.dump(annotation, annotation_file, indent=2, sort_keys=False)

    total_time_taken = time.time() - total_start_time

    print 'Total analysis time = %s' % disp_time(total_time_taken)

if __name__ == "__main__":

    main(CLIArgumentParser().parse_args())
