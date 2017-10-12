from scipy import ndimage
from skimage.measure import regionprops
from skimage.segmentation import slic, find_boundaries
from keras.models import Model, load_model

import os
import sys
import json
import time

import numpy as np
import dask
import h5py

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

    # get current magnification
    magnification = ts.getMetadata()['magnification']

    # get requested tile
    tile_info = ts.getSingleTile(
        tile_position=tile_position,
        format=large_image.tilesource.TILE_FORMAT_NUMPY,
        **it_kwargs)

    # get tile image
    im_tile = tile_info['tile'][:, :, :3]

    # get current left and top positions
    tile_left_position = tile_info['gx']
    tile_top_position = tile_info['gy']

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

    # detect real stains
    dict_stains = {}

    # compare w with stain_color_map
    w = w.T

    for i in range(w.shape[0]):
        for j in htk_cdeconv.stain_color_map:
            if len(set(w[i]) & set(htk_cdeconv.stain_color_map[j])) == 3:
                dict_stains[j] = i

    # compute the number of super-pixels
    im_width, im_height = im_tile.shape[:2]
    n_superpixels = (im_width/args.patchSize) * (im_height/args.patchSize)

    #
    # Generate labels using a superpixel algorithm (SLIC)
    # In SLIC, compactness controls image space proximity.
    # Higher compactness will make the shape of superpixels more square.
    #
    im_label = slic(im_nmzd, n_segments=n_superpixels,
                    compactness=args.compactness) + 1

    region_props = regionprops(im_label)

    n_labels = len(region_props)

    # set superpixel data array
    s_data = np.zeros((
        n_labels, args.patchSize, args.patchSize, len(dict_stains))
    )

    # set x, y centroids array
    x_cent = np.zeros((n_labels, 1), dtype=np.float32)
    y_cent = np.zeros((n_labels, 1), dtype=np.float32)

    # set x, y boundary list
    x_brs = []
    y_brs = []

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

            # find boundaries
            label_boundary = np.argwhere(
                find_boundaries(lmask, mode="inner").astype(np.uint8) == 1)

            # get superpixel boundary at highest-res
            x_boundary = label_boundary[:, 0] * (magnification / args.analysis_mag) + \
                tile_left_position

            y_boundary = label_boundary[:, 1] * (magnification / args.analysis_mag) + \
                tile_top_position

            x_brs.append(x_boundary)
            y_brs.append(y_boundary)

            # get superpixel centers at highest-res
            x_cent[t_num] = cen_x * (magnification / args.analysis_mag) + \
                tile_left_position
            y_cent[t_num] = cen_y * (magnification / args.analysis_mag) + \
                tile_top_position

            # increase true label number by 1
            t_num = t_num + 1

    s_data_out = s_data[:t_num, :, :, :]
    x_cent_out = x_cent[:t_num]
    y_cent_out = y_cent[:t_num]

    return s_data_out, x_cent_out, y_cent_out, x_brs, y_brs


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

    if os.path.splitext(args.outputSuperpixelFeatureFile)[1] not in ['.h5']:
        raise ValueError('Extension of output feature file must be .h5')


def main(args):  # noqa: C901

    total_start_time = time.time()

    print('\n>> CLI Parameters ...\n')

    print args

    #
    # Check whether slide path is a file or a directory
    #
    if os.path.isfile(args.inputSlidePath):
        img_paths = [args.inputSlidePath]

    elif os.path.isdir(args.inputSlidePath):
        img_paths = [
            os.path.join(args.inputSlidePath, files)
            for files in os.listdir(args.inputSlidePath)
            if os.path.isfile(
                os.path.join(args.inputSlidePath, files))]

    else:
        raise IOError('Slide path does not exist.')

    check_args(args)

    feature_file_format = os.path.splitext(args.outputSuperpixelFeatureFile)[1]

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

    n_images = len(img_paths)

    slide_name_list = []
    superpixel_index = 0
    first_superpixel_index = np.zeros((n_images, 1), dtype=np.int32)
    slide_superpixel_index = []

    slide_superpixel_data = []
    slide_x_centroids = []
    slide_y_centroids = []
    slide_x_boundaries = []
    slide_y_boundaries = []

    for i in range(n_images):

        #
        # Read Input Image
        #
        print('\n>> Reading input image ... \n')

        ts = large_image.getTileSource(img_paths[i])

        ts_metadata = ts.getMetadata()

        print json.dumps(ts_metadata, indent=2)

        is_wsi = ts_metadata['magnification'] is not None

        #
        # Compute tissue/foreground mask at low-res for whole slide images
        #
        if is_wsi:

            print('\n>> Computing tissue/foreground mask at low-res ...\n')

            im_fgnd_mask_lres, fgnd_seg_scale = \
                cli_utils.segment_wsi_foreground_at_low_res(ts)

        #
        # Compute foreground fraction of tiles in parallel using Dask
        #
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
                img_paths[i], im_fgnd_mask_lres, fgnd_seg_scale,
                **it_kwargs
            )

            num_fgnd_tiles = np.count_nonzero(
                tile_fgnd_frac_list >= args.min_fgnd_frac)

            percent_fgnd_tiles = 100.0 * num_fgnd_tiles / num_tiles

            fgnd_frac_comp_time = time.time() - start_time

            print 'Number of foreground tiles = %d (%.2f%%)' % (
                num_fgnd_tiles, percent_fgnd_tiles)

            print 'Time taken = %s' % cli_utils.disp_time_hms(fgnd_frac_comp_time)

        #
        # Detect superpixel data in parallel using Dask
        #
        print('\n>> Detecting superpixel data ...\n')

        start_time = time.time()

        tile_result_list = []

        for tile in ts.tileIterator(**it_kwargs):

            tile_position = tile['tile_position']['position']

            if is_wsi and tile_fgnd_frac_list[tile_position] <= args.min_fgnd_frac:
                continue

            # detect superpixel data
            cur_result = dask.delayed(compute_superpixel_data)(
                img_paths[i],
                tile_position,
                args, **it_kwargs)

            # append result to list
            tile_result_list.append(cur_result)

        tile_result_list = dask.delayed(tile_result_list).compute()

        print('\n>> tile_result_list to slide information ...\n')

        superpixel_data = np.asarray([
            data for s_data_out, x_cent_out, y_cent_out, x_brs, y_brs
            in tile_result_list for data in s_data_out]
        )

        x_centroids = np.asarray([
            centers for s_data_out, x_cent_out, y_cent_out, x_brs, y_brs
            in tile_result_list for centers in x_cent_out]
        )

        y_centroids = np.asarray([
            centers for s_data_out, x_cent_out, y_cent_out, x_brs, y_brs
            in tile_result_list for centers in y_cent_out]
        )

        x_boundaries = [
            boundary for s_data_out, x_cent_out, y_cent_out, x_brs, y_brs
            in tile_result_list for boundary in x_brs]

        y_boundaries = [
            boundary for s_data_out, x_cent_out, y_cent_out, x_brs, y_brs
            in tile_result_list for boundary in y_brs]

        #
        # Get slide information, superpixel data, and x, y centroids
        #
        n_superpixels = len(superpixel_data)

        base = os.path.basename(img_paths[i])

        # get slide name
        slide_name_list.append(os.path.splitext(base)[0])

        # get first superpixel index
        first_superpixel_index[i, 0] = superpixel_index

        # get slide index for superpixel
        slide_index = np.zeros((n_superpixels, 1), dtype=np.int32)
        slide_index.fill(i)

        if i == 0:

            slide_superpixel_index = slide_index
            slide_superpixel_data = superpixel_data
            slide_x_centroids = x_centroids
            slide_y_centroids = y_centroids
            slide_x_boundaries = x_boundaries
            slide_y_boundaries = y_boundaries

        else:

            slide_superpixel_index = np.append(
                slide_superpixel_index, slide_index, axis=0)
            slide_superpixel_data = np.append(
                slide_superpixel_data, superpixel_data, axis=0)
            slide_x_centroids = np.append(slide_x_centroids, x_centroids, axis=0)
            slide_y_centroids = np.append(slide_y_centroids, y_centroids, axis=0)

            for x in x_boundaries:
                slide_x_boundaries.append(x)
            for y in y_boundaries:
                slide_y_boundaries.append(y)

        superpixel_detection_time = time.time() - start_time

        print "Superpixel detection time taken = %s" % \
              cli_utils.disp_time_hms(superpixel_detection_time)

        superpixel_index = superpixel_index + len(superpixel_data)

    #
    # Compute superpixel data based on the input model
    #
    print('\n>> Compute superpixel data to extract low dimensional features ...\n')
    start_time = time.time()

    # load input model
    model = load_model(args.inputModelFile)

    encoded_layer_out = 0

    encoded_layer_num = -1

    layer_num_temp = -1

    # loop layers in the input model
    for layer in model.layers:

        # find encoding layer number assuming that the shape of encoding layer is 2
        # and the encoding layer has the lowest dimension
        if len(layer.output.shape) == 2 and encoded_layer_out < layer.output.shape[1]:
            encoded_layer_out = layer.output.shape[1]
            encoded_layer_num = layer_num_temp

        # decrease layer number by -1
        layer_num_temp = layer_num_temp - 1

    encoder = Model(
        inputs=model.input, outputs=model.layers[encoded_layer_num].output)

    # predict superpixel features using encoder
    encoded_features = encoder.predict(slide_superpixel_data)

    encoding_time = time.time() - start_time

    print 'Encoding time = %s' % cli_utils.disp_time_hms(encoding_time)

    # get encoder size from the last layer of the input model
    encoder_size = encoder.layers[-1].get_output_at(0).get_shape().as_list()[1]

    # get mean and standard deviation
    slide_feature_mean = np.reshape(
        np.mean(encoded_features, axis=0), (encoder_size, 1)
    ).astype(np.float32)

    slide_feature_stddev = np.reshape(
        np.std(encoded_features, axis=0), (encoder_size, 1)
    ).astype(np.float32)

    #
    # Create Slide Data file
    #
    print('>> Writing H5 data file')

    if feature_file_format == '.h5':

        output = h5py.File(args.outputSuperpixelFeatureFile, 'w')
        output.create_dataset('slides', data=slide_name_list)
        output.create_dataset('slideIdx', data=slide_superpixel_index)
        output.create_dataset('dataIdx', data=first_superpixel_index)
        output.create_dataset('mean', data=slide_feature_mean)
        output.create_dataset('std_dev', data=slide_feature_stddev)
        output.create_dataset('features', data=encoded_features)
        output.create_dataset('x_centroid', data=slide_x_centroids)
        output.create_dataset('y_centroid', data=slide_y_centroids)
        output.close()

    else:

        raise ValueError('Extension of output data file must be .h5')

    #
    # Create Text file for boundaries
    #
    print('>> Writing text boundary file')

    boundary_file = open(args.outputBoundariesFile, 'w')
    boundary_file.write("Slide\tX\tY\tBoundaries: x1,y1;x2,y2;...\n")
    for i in range(superpixel_index):
        boundary_file.write("%s\t" % slide_name_list[slide_superpixel_index[i, 0]])
        boundary_file.write("%f\t" % slide_x_centroids[i, 0])
        boundary_file.write("%f\t" % slide_y_centroids[i, 0])
        for j in range(len(slide_x_boundaries[i])):
            boundary_file.write(
                "%d,%d " % (slide_x_boundaries[i][j], slide_y_boundaries[i][j]))
        boundary_file.write("\n")

    total_time_taken = time.time() - total_start_time

    print 'Total analysis time = %s' % cli_utils.disp_time_hms(total_time_taken)


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
