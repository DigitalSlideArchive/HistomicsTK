from scipy import ndimage
from skimage.measure import regionprops
from skimage.segmentation import slic
from skimage.transform import resize
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
import histomicstk.segmentation as htk_seg
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

    # get global magnification
    gmagnification = ts.getMetadata()['magnification']

    # get requested tile information
    tile_info = ts.getSingleTile(
        tile_position=tile_position,
        format=large_image.tilesource.TILE_FORMAT_NUMPY,
        **it_kwargs)

    # get global x and y position
    left = tile_info['gx']
    top = tile_info['gy']

    # get current magnification
    magnification = tile_info['magnification']

    # get magnification ratio from analysis_mag
    magnification_ratio = magnification / args.analysis_mag

    im_tile_org = tile_info['tile'][:, :, :3]

    im_tile = resize(
        im_tile_org, (im_tile_org.shape[0] / magnification_ratio,
                      im_tile_org.shape[1] / magnification_ratio),
        mode='reflect')

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

    # set superpixel data list
    s_data = []
    x_cent = []
    y_cent = []
    x_brs = []
    y_brs = []

    for i in range(n_labels):
        # get x, y centroids for superpixel
        cen_x, cen_y = region_props[i].centroid

        # get bounds of superpixel region
        min_row, max_row, min_col, max_col = \
            get_patch_bounds(cen_x, cen_y, args.patchSize, im_width, im_height)

        # grab superpixel label mask
        lmask = (
            im_label[:, :] == region_props[i].label).astype(np.bool)

        # embed with center pixel in middle of padded window
        emask = np.zeros((lmask.shape[0] + 2, lmask.shape[1] + 2), dtype=np.bool)
        emask[1:-1, 1:-1] = lmask

        if np.sum(im_fgnd_mask & lmask) > args.min_fgnd_superpixel:
            # get variance of superpixel region
            var = ndimage.variance(
                im_nmzd[min_row:max_row, min_col:max_col, :] / 255.0)

            if var < args.min_var_superpixel:
                continue

            # get superpixel data
            stain_data = np.zeros(
                (args.patchSize, args.patchSize, len(dict_stains)))

            for key in dict_stains.keys():
                k = dict_stains[key]
                im_stain = im_stains[:, :, k].astype(np.float) / 255.0
                stain_data[:, :, k] = im_stain[min_row:max_row, min_col:max_col]

            s_data.append(stain_data)

            # find boundaries
            bx, by = htk_seg.label.trace_object_boundaries(emask)

            with np.errstate(invalid='ignore'):
                # remove redundant points
                mby, mbx = htk_utils.merge_colinear(by[0].astype(float), bx[0].astype(float))

            # get superpixel boundary at highest-res
            x_brs.append(
                (mbx - 1) * (gmagnification / args.analysis_mag) + top)
            y_brs.append(
                (mby - 1) * (gmagnification / args.analysis_mag) + left)

            # get superpixel centers at highest-res
            x_cent.append(
                cen_x * (gmagnification / args.analysis_mag) + top)
            y_cent.append(
                cen_y * (gmagnification / args.analysis_mag) + left)

    return s_data, x_cent, y_cent, x_brs, y_brs


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

        superpixel_data = []
        x_centroids = []
        y_centroids = []
        x_boundaries = []
        y_boundaries = []

        for s_data, x_cent, y_cent, x_brs, y_brs in tile_result_list:

            for s_d in s_data:
                superpixel_data.append(s_d)

            for x_c in x_cent:
                x_centroids.append(x_c)

            for y_c in y_cent:
                y_centroids.append(y_c)

            for x_b in x_brs:
                x_boundaries.append(x_b)

            for y_b in y_brs:
                y_boundaries.append(y_b)

        superpixel_data = np.asarray(superpixel_data, dtype=np.float32)

        n_superpixels = len(superpixel_data)

        x_centroids = np.round(
            np.asarray(x_centroids, dtype=np.float32).reshape((n_superpixels, 1)),
            decimals=1)

        y_centroids = np.round(
            np.asarray(y_centroids, dtype=np.float32).reshape((n_superpixels, 1)),
            decimals=1)

        # get slide name
        base = os.path.basename(img_paths[i])
        s_name = os.path.splitext(base)[0].split(".")
        slide_name_list.append(s_name[0])

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

        # find encoding layer
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
        output.create_dataset('x_centroid', data=slide_y_centroids)
        output.create_dataset('y_centroid', data=slide_x_centroids)
        output.create_dataset('patch_size', data=args.patchSize)
        output.close()

    else:

        raise ValueError('Extension of output data file must be .h5')

    #
    # Create Text file for boundaries
    #
    print('>> Writing text boundary file')

    boundary_file = open(args.outputBoundariesFile, 'w')

    for i in range(superpixel_index):
        boundary_file.write("%s\t" % slide_name_list[slide_superpixel_index[i, 0]])
        boundary_file.write("%f\t" % slide_y_centroids[i, 0])
        boundary_file.write("%f\t" % slide_x_centroids[i, 0])

        for j in range(len(slide_x_boundaries[i])):
            boundary_file.write(
                "%d,%d " % (slide_y_boundaries[i][j], slide_x_boundaries[i][j]))

        boundary_file.write("\n")

    total_time_taken = time.time() - total_start_time

    print 'Total analysis time = %s' % cli_utils.disp_time_hms(total_time_taken)


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
