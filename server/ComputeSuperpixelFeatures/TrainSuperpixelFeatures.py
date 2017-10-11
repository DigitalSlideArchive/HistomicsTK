from scipy import ndimage
from skimage.measure import regionprops
from skimage.segmentation import slic
from keras.models import load_model

import os
import sys
import json
import time

import numpy as np
import dask

import histomicstk.preprocessing.color_normalization as htk_cnorm
import histomicstk.preprocessing.color_deconvolution as htk_cdeconv
import histomicstk.utils as htk_utils

import large_image

from ctk_cli import CLIArgumentParser

import logging
logging.basicConfig(level=logging.CRITICAL)

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), '..')))
from cli_common import utils as cli_utils  # noqa


def compute_superpixel_data(img_path, tile_position, args, **it_kwargs):

    # get slide tile source
    ts = large_image.getTileSource(img_path)

    # get requested tile
    tile_info = ts.getSingleTile(
        tile_position=tile_position,
        format=large_image.tilesource.TILE_FORMAT_NUMPY,
        **it_kwargs)

    # get tile image
    im_tile = tile_info['tile'][:, :, :3]

    # perform color normalization
    im_nmzd = htk_cnorm.reinhard(im_tile,
                                 args.reference_mu_lab,
                                 args.reference_std_lab)

    # get red and green channels
    im_red = im_nmzd[:, :, 0]
    im_green = im_nmzd[:, :, 1]

    # get foreground mask simply using numpy.spacing, a generalization of EPS
    im_ratio = im_red / (im_green + np.spacing(1))
    im_fgnd_mask = im_ratio > args.rg_ratio_superpixel

    # perform color decovolution
    w = cli_utils.get_stain_matrix(args)

    im_stains = htk_cdeconv.color_deconvolution(im_nmzd, w).Stains

    #
    # Detect real stains
    #
    dict_stains = {}

    # compare w with stain_color_map
    w = w.T

    for i in range(w.shape[0]):
        for j in htk_cdeconv.stain_color_map:
            if len(set(w[i])&set(htk_cdeconv.stain_color_map[j])) == 3:
                dict_stains[j] = i

    # compute number of super-pixels
    im_width, im_height = im_tile.shape[:2]
    n_superpixels = (im_width/args.patchSize) * (im_height/args.patchSize)

    #
    # Perform a superpixel algorithm (SLIC)
    # In SLIC, compactness controls image space proximity.
    # Higher compactness will make the shape of superpixels more square.
    #
    im_label = slic(im_nmzd, n_segments=n_superpixels,
                    compactness=args.compactness) + 1

    # extract features
    region_props = regionprops(im_label)

    n_labels = len(region_props)

    # set superpixel data array
    s_data = np.zeros((
        n_labels, args.patchSize, args.patchSize, len(dict_stains)))

    # set true label number. labels will be removed by foreground mask
    t_num = 0

    for i in range(n_labels):

        # get x, y centroids for superpixel
        cen_x, cen_y = region_props[i].centroid

        # get bounds of superpixel region
        min_row, max_row, min_col, max_col = \
            get_patch_bounds(cen_x, cen_y, args.patchSize, im_width, im_height)

        # grab superpixel label mask
        lmask = (
            im_label[:, :] == region_props[i].label).astype(np.bool)

        if np.sum(im_fgnd_mask & lmask) > args.min_fgnd_superpixel:

            # get variance of superpixel region
            var = ndimage.variance(
                im_nmzd[min_row:max_row, min_col:max_col, :] / 255.0)

            if var < args.min_var_superpixel:
                continue

            # get superpixel data
            for key in dict_stains.keys():
                k = dict_stains[key]
                im_stain = im_stains[:, :, k].astype(np.float) / 255.0
                s_data[t_num, :, :, k] = im_stain[min_row:max_row, min_col:max_col]

            # increase true label number by 1
            t_num = t_num + 1

    s_data_out = s_data[:t_num, :, :, :]

    return s_data_out


def create_train_model(input_data, input_model, args):

    # load input model
    autoencoder = load_model(input_model)

    # fit to autoencoder
    autoencoder.fit(
        input_data, input_data, epochs=args.epochs, batch_size=args.batchSize)

    return autoencoder


def get_patch_bounds(cx, cy, patch_size, m, n):

    half_patch_size = patch_size/2.0

    min_row = int(round(cx) - half_patch_size)
    max_row = int(round(cx) + half_patch_size)
    min_col = int(round(cy) - half_patch_size)
    max_col = int(round(cy) + half_patch_size)

    if min_row < 0:
        max_row = max_row - min_row
        min_row = 0

    if max_row > m-1:
        min_row = min_row - (max_row - (m-1))
        max_row = m-1

    if min_col < 0:
        max_col = max_col - min_col
        min_col = 0

    if max_col > n-1:
        min_col = min_col - (max_col - (n-1))
        max_col = n-1

    return min_row, max_row, min_col, max_col


def check_args(args):

    if len(args.reference_mu_lab) != 3:
        raise ValueError('Reference Mean LAB should be a 3 element vector.')

    if len(args.reference_std_lab) != 3:
        raise ValueError('Reference Stddev LAB should be a 3 element vector.')

    if len(args.analysis_roi) != 4:
        raise ValueError('Analysis ROI must be a vector of 4 elements.')

    if os.path.splitext(args.outputModelFile)[1] not in ['.h5']:
        raise ValueError('Extension of output feature file must be .h5')


def main(args):

    total_start_time = time.time()

    print('\n>> CLI Parameters ...\n')

    print args

    #
    # Check whether slide path is a file or a directory
    #
    if os.path.isfile(args.inputSlidePath):
        img_path = [args.inputSlidePath]

    elif os.path.isdir(args.inputSlidePath):
        img_path = [
            os.path.join(args.inputSlidePath, files)
            for files in os.listdir(args.inputSlidePath)
            if os.path.isfile(
                os.path.join(args.inputSlidePath, files))]

    else:
        raise IOError('Slide path does not exist.')

    check_args(args)

    feature_file_format = os.path.splitext(args.outputModelFile)[1]

    if np.all(np.array(args.analysis_roi) == -1):
        process_whole_image = True
    else:
        process_whole_image = False

    #
    # Initiate Dask client
    #
    print('\n>> Creating Dask client ...\n')

    c = cli_utils.create_dask_client(args)

    print c

    #
    # Extract superpixel data to be trained
    #
    slide_superpixel_data = []

    for i in range(len(img_path)):

        # read input image
        print('\n>> Reading input image ... \n')

        ts = large_image.getTileSource(img_path[i])

        ts_metadata = ts.getMetadata()

        print json.dumps(ts_metadata, indent=2)

        is_wsi = ts_metadata['magnification'] is not None

        # compute tissue/foreground mask at low-res for whole slide images
        if is_wsi:

            print('\n>> Computing tissue/foreground mask at low-res ...\n')

            im_fgnd_mask_lres, fgnd_seg_scale = \
                cli_utils.segment_wsi_foreground_at_low_res(ts)

        # compute foreground fraction of tiles in parallel using Dask
        tile_fgnd_frac_list = [1.0]

        it_kwargs = {
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

            tile_fgnd_frac_list = htk_utils.compute_tile_foreground_fraction(
                img_path[i], im_fgnd_mask_lres, fgnd_seg_scale,
                **it_kwargs
            )

            num_fgnd_tiles = np.count_nonzero(
                tile_fgnd_frac_list >= args.min_fgnd_frac)

            percent_fgnd_tiles = 100.0 * num_fgnd_tiles / num_tiles

            fgnd_frac_comp_time = time.time() - start_time

            print 'Number of foreground tiles = %d (%.2f%%)' % (
                num_fgnd_tiles, percent_fgnd_tiles)

            print 'Time taken = %s' % cli_utils.disp_time_hms(fgnd_frac_comp_time)

        # detect superpixel data in parallel using Dask
        print('\n>> Detecting superpixel data ...\n')

        start_time = time.time()

        tile_result_list = []

        for tile in ts.tileIterator(**it_kwargs):

            tile_position = tile['tile_position']['position']

            if is_wsi and tile_fgnd_frac_list[tile_position] <= args.min_fgnd_frac:
                continue

            # detect superpixel data
            cur_result = dask.delayed(compute_superpixel_data)(
                img_path[i],
                tile_position,
                args, **it_kwargs)

            # append result to list
            tile_result_list.append(cur_result)

        tile_result_list = dask.delayed(tile_result_list).compute()

        print('\n>> tile_result_list to superpixel data ...\n')

        superpixel_data = np.asarray([data for s_data_out in tile_result_list
                                      for data in s_data_out])

        if i == 0:

            slide_superpixel_data = superpixel_data

        else:

            slide_superpixel_data = np.append(
                slide_superpixel_data, superpixel_data, axis=0)

        superpixel_detection_time = time.time() - start_time

        print "Superpixel detection time taken = %s" % \
              cli_utils.disp_time_hms(superpixel_detection_time)

    #
    # Train input model
    #
    print('\n>> Train input model ...\n')

    start_time = time.time()

    model = create_train_model(slide_superpixel_data, args.inputModelFile, args)

    training_time = time.time() - start_time

    print 'Training time = %s' % cli_utils.disp_time_hms(training_time)

    #
    # Create a model file
    #
    print('>> Writing H5 model file')

    if feature_file_format == '.h5':

        model.save(args.outputModelFile)

    else:

        raise ValueError('Extension of output feature file must be .h5')

    total_time_taken = time.time() - total_start_time

    print 'Total analysis time = %s' % cli_utils.disp_time_hms(total_time_taken)


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
