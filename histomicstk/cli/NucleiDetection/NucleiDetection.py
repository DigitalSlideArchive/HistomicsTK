import json
import logging
import os
import time

import large_image
import numpy as np

import histomicstk.preprocessing.color_deconvolution as htk_cdeconv
import histomicstk.preprocessing.color_normalization as htk_cnorm
import histomicstk.segmentation.label as htk_seg_label
import histomicstk.segmentation.nuclear as htk_nuclear
import histomicstk.utils as htk_utils
from histomicstk.cli import utils as cli_utils
from histomicstk.cli.utils import CLIArgumentParser

logging.basicConfig(level=logging.CRITICAL)


def detect_tile_nuclei(slide_path, tile_position, args, it_kwargs,
                       src_mu_lab=None, src_sigma_lab=None):

    # get slide tile source
    ts = large_image.getTileSource(slide_path)

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
                                 args.reference_std_lab,
                                 src_mu=src_mu_lab,
                                 src_sigma=src_sigma_lab)

    # perform color decovolution
    w = cli_utils.get_stain_matrix(args)

    im_stains = htk_cdeconv.color_deconvolution(im_nmzd, w).Stains

    im_nuclei_stain = im_stains[:, :, 0].astype(float)

    # segment nuclear foreground
    im_nuclei_fgnd_mask = im_nuclei_stain < args.foreground_threshold

    # segment nuclei
    im_nuclei_seg_mask = htk_nuclear.detect_nuclei_kofahi(
        im_nuclei_stain,
        im_nuclei_fgnd_mask,
        args.min_radius,
        args.max_radius,
        args.min_nucleus_area,
        args.local_max_search_radius
    )

    # Delete border nuclei
    if args.ignore_border_nuclei is True:

        im_nuclei_seg_mask = htk_seg_label.delete_border(im_nuclei_seg_mask)

    # generate nuclei annotations
    nuclei_annot_list = []

    flag_nuclei_found = np.any(im_nuclei_seg_mask)

    if flag_nuclei_found:
        nuclei_annot_list = cli_utils.create_tile_nuclei_annotations(
            im_nuclei_seg_mask, tile_info, args.nuclei_annotation_format)

    return nuclei_annot_list


def main(args):
    import dask

    total_start_time = time.time()

    print('\n>> CLI Parameters ...\n')

    print(args)

    if not os.path.isfile(args.inputImageFile):
        raise OSError('Input image file does not exist.')

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

    start_time = time.time()

    c = cli_utils.create_dask_client(args)

    print(c)

    dask_setup_time = time.time() - start_time
    print('Dask setup time = {}'.format(
        cli_utils.disp_time_hms(dask_setup_time)))

    #
    # Read Input Image
    #
    print('\n>> Reading input image ... \n')

    ts = large_image.getTileSource(args.inputImageFile)

    ts_metadata = ts.getMetadata()

    print(json.dumps(ts_metadata, indent=2))

    is_wsi = ts_metadata['magnification'] is not None

    #
    # Compute tissue/foreground mask at low-res for whole slide images
    #
    if is_wsi and process_whole_image:

        print('\n>> Computing tissue/foreground mask at low-res ...\n')

        start_time = time.time()

        im_fgnd_mask_lres, fgnd_seg_scale = \
            cli_utils.segment_wsi_foreground_at_low_res(ts)

        fgnd_time = time.time() - start_time

        print('low-res foreground mask computation time = {}'.format(
            cli_utils.disp_time_hms(fgnd_time)))

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

        print(f'Number of tiles = {num_tiles}')

        if process_whole_image:

            tile_fgnd_frac_list = htk_utils.compute_tile_foreground_fraction(
                args.inputImageFile, im_fgnd_mask_lres, fgnd_seg_scale,
                it_kwargs
            )

        else:

            tile_fgnd_frac_list = np.full(num_tiles, 1.0)

        num_fgnd_tiles = np.count_nonzero(
            tile_fgnd_frac_list >= args.min_fgnd_frac)

        percent_fgnd_tiles = 100.0 * num_fgnd_tiles / num_tiles

        fgnd_frac_comp_time = time.time() - start_time

        print('Number of foreground tiles = {:d} ({:2f}%%)'.format(
            num_fgnd_tiles, percent_fgnd_tiles))

        print('Tile foreground fraction computation time = {}'.format(
            cli_utils.disp_time_hms(fgnd_frac_comp_time)))

    #
    # Compute reinhard stats for color normalization
    #
    src_mu_lab = None
    src_sigma_lab = None

    if is_wsi and process_whole_image:

        print('\n>> Computing reinhard color normalization stats ...\n')

        start_time = time.time()

        src_mu_lab, src_sigma_lab = htk_cnorm.reinhard_stats(
            args.inputImageFile, 0.01, magnification=args.analysis_mag)

        rstats_time = time.time() - start_time

        print('Reinhard stats computation time = {}'.format(
            cli_utils.disp_time_hms(rstats_time)))

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
            args, it_kwargs,
            src_mu_lab, src_sigma_lab
        )

        # append result to list
        tile_nuclei_list.append(cur_nuclei_list)

    tile_nuclei_list = dask.delayed(tile_nuclei_list).compute()

    nuclei_list = [anot
                   for anot_list in tile_nuclei_list for anot in anot_list]

    nuclei_detection_time = time.time() - start_time

    print('Number of nuclei = {}'.format(len(nuclei_list)))

    print('Nuclei detection time = {}'.format(
        cli_utils.disp_time_hms(nuclei_detection_time)))

    #
    # Write annotation file
    #
    print('\n>> Writing annotation file ...\n')

    annot_fname = os.path.splitext(
        os.path.basename(args.outputNucleiAnnotationFile))[0]

    annotation = {
        "name":     annot_fname + '-nuclei-' + args.nuclei_annotation_format,
        "elements": nuclei_list
    }

    with open(args.outputNucleiAnnotationFile, 'w') as annotation_file:
        json.dump(annotation, annotation_file, indent=2, sort_keys=False)

    total_time_taken = time.time() - total_start_time

    print('Total analysis time = {}'.format(
        cli_utils.disp_time_hms(total_time_taken)))


if __name__ == "__main__":

    main(CLIArgumentParser().parse_args())
