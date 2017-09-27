from skimage.measure import regionprops
from skimage.segmentation import slic
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


def compute_input_tensors(slide_path, tile_position, args, **it_kwargs):

    # get slide tile source
    ts = large_image.getTileSource(slide_path)

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

    # perform color decovolution
    w = cli_utils.get_stain_matrix(args)

    im_stains = htk_cdeconv.color_deconvolution(im_nmzd, w).Stains

    im_hematoxylin_stain = im_stains[:, :, 0].astype(np.float)/255.0
    im_eosin_stain = im_stains[:, :, 1].astype(np.float)/255.0

    # get the number of super-pixels
    im_width, im_height = im_tile.shape[:2]
    n_superpixels = (im_width/args.patchSize) * (im_height/args.patchSize)

    # perform SLIC to get labels with max compactness
    # slic has a value 0 as a first label, so added 1
    im_label = slic(im_nmzd, n_segments=n_superpixels, compactness=100) + 1

    # extract features
    region_props = regionprops(im_label)

    # numFeatures from hematoxylin and eosin stained images
    n_labels = len(region_props)

    # we will use feature shape: NHWC on CPU, NCHW on GPU
    # with 2 channels: hematoxylin adn eosin
    fdata = np.zeros((n_labels, args.patchSize, args.patchSize, 2))

    # set an array for x, y centers
    cdata = np.zeros((n_labels, 2))

    for i in range(n_labels):
        cen_x, cen_y = region_props[i].centroid

        # get bounds of region
        min_row, max_row, min_col, max_col = \
            get_patch_bounds(cen_x, cen_y, args.patchSize, im_width, im_height)

        # get image centers at low-res
        cdata[i, 0] = cen_x * (magnification / args.analysis_mag) + \
            tile_left_position
        cdata[i, 1] = cen_y * (magnification / args.analysis_mag) + \
            tile_top_position

        # get image patches
        fdata[i, :, :, 0] = im_hematoxylin_stain[min_row:max_row, min_col:max_col]
        fdata[i, :, :, 1] = im_eosin_stain[min_row:max_row, min_col:max_col]

    return cdata, fdata


def get_encoded_features(input_tensors, input_model, args):

    # load input model
    autoencoder = load_model(input_model)

    # input: the first input of the model, output: the last output of the model
    encoder = Model(inputs=autoencoder.layers[0].input, outputs=autoencoder.layers[6].output)

    # fit to autoencoder with the number of epochs and the batch size
    autoencoder.fit(input_tensors, input_tensors, epochs=args.epochs, batch_size=args.batchSize,
                    shuffle=True, validation_data=(input_tensors, input_tensors))

    # extract encoded features
    features = encoder.predict(input_tensors)

    return features


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

    if not os.path.isfile(args.inputImageFile):
        raise IOError('Input image file does not exist.')

    if len(args.reference_mu_lab) != 3:
        raise ValueError('Reference Mean LAB should be a 3 element vector.')

    if len(args.reference_std_lab) != 3:
        raise ValueError('Reference Stddev LAB should be a 3 element vector.')

    if len(args.analysis_roi) != 4:
        raise ValueError('Analysis ROI must be a vector of 4 elements.')

    if os.path.splitext(args.outputSuperpixelFeatureFile)[1] not in ['.h5']:
        raise ValueError('Extension of output feature file must be .h5')


def main(args):

    total_start_time = time.time()

    print('\n>> CLI Parameters ...\n')

    print args

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
            args.inputImageFile, im_fgnd_mask_lres, fgnd_seg_scale,
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
    # Detect and compute nuclei features in parallel using Dask
    #
    print('\n>> Detecting superpixels and computing features ...\n')

    start_time = time.time()

    tile_result_list = []

    n_tiles = 0

    for tile in ts.tileIterator(**it_kwargs):

        tile_position = tile['tile_position']['position']

        if is_wsi and tile_fgnd_frac_list[tile_position] <= args.min_fgnd_frac:
            continue

        n_tiles = n_tiles + 1

        # detect image patches
        cur_result = dask.delayed(compute_input_tensors)(
            args.inputImageFile,
            tile_position,
            args, **it_kwargs)

        # append result to list
        tile_result_list.append(cur_result)

    tile_result_list = dask.delayed(tile_result_list).compute()

    print 'Number of tiles processed = %d' % n_tiles
    print('\n>> tile_result_list to centers and features ...\n')

    slide_centers = np.asarray([centers
                                for cdata, fdata in tile_result_list
                                for centers in cdata])

    slide_pixel_array = np.asarray([features
                                    for cdata, fdata in tile_result_list
                                    for features in fdata])

    patch_detection_time = time.time() - start_time

    print 'Number of image patches = ', len(slide_centers)
    print "Patch detection time taken = %s" % cli_utils.disp_time_hms(patch_detection_time)

    #
    # Fit to autoencoder model to extract the encoded features
    #
    print('\n>> Training model to extract low dimensional features ...\n')

    start_time = time.time()

    encoded_features = get_encoded_features(slide_pixel_array, args.inputModelFile, args)

    encoding_time = time.time() - start_time

    print 'Encoding time = %s' % cli_utils.disp_time_hms(encoding_time)

    #
    # Get slide index, slide patch index, slide name, mean, standard deviation, and x, y centers
    #
    slide_patch_num = len(slide_centers)
    slide_num = 1

    # get slide names
    slide_names = []
    base = os.path.basename(args.inputImageFile)
    slide_name = os.path.splitext(base)[0]
    slide_names = np.append(slide_names, slide_name)

    # get slide index for each image patch
    slide_index = np.zeros((slide_patch_num, 1), dtype=np.int)
    for i in range(slide_patch_num):
        slide_index[i, 0] = i

    # get index of the first patch for each slide
    slide_patch_index = np.zeros((slide_num, 1), dtype=np.int)
    for i in range(slide_num):
        slide_patch_index[i, 0] = 0

    # get mean and standard deviation for each feature
    slide_feature_num = encoded_features.shape[1]
    slide_feature_mean = np.zeros((slide_feature_num, 1), dtype=np.float32)
    slide_feature_stddev = np.zeros((slide_feature_num, 1), dtype=np.float32)
    slide_feature_mean[:, 0] = np.mean(encoded_features, axis=0)
    slide_feature_stddev[:, 0] = np.std(encoded_features, axis=0)

    # get x, y centers for each image patch
    slide_center_x_list = np.zeros((slide_patch_num, 1), dtype=np.float32)
    slide_center_y_list = np.zeros((slide_patch_num, 1), dtype=np.float32)
    slide_center_x_list[:, 0] = slide_centers[:, 0]
    slide_center_y_list[:, 0] = slide_centers[:, 1]

    #
    # Create Feature file
    #
    print('>> Writing H5 feature file')

    if feature_file_format == '.h5':

        output = h5py.File(args.outputSuperpixelFeatureFile, 'w')
        output.create_dataset('slides', data=slide_names)
        output.create_dataset('slideIdx', data=slide_index)
        output.create_dataset('dataIdx', data=slide_patch_index)
        output.create_dataset('mean', data=slide_feature_mean)
        output.create_dataset('std_dev', data=slide_feature_stddev)
        output.create_dataset('features', data=encoded_features)
        output.create_dataset('x_centroid', data=slide_center_x_list)
        output.create_dataset('y_centroid', data=slide_center_y_list)
        output.close()

    else:

        raise ValueError('Extension of output feature file must be .h5')

    total_time_taken = time.time() - total_start_time

    print 'Total analysis time = %s' % cli_utils.disp_time_hms(total_time_taken)


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
