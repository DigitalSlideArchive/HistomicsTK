import json
import logging
import os
import time
from pathlib import Path

import large_image
import numpy as np

import histomicstk
import histomicstk.preprocessing.color_normalization as htk_cnorm
import histomicstk.segmentation.label as htk_seg_label
import histomicstk.segmentation.nuclear as htk_nuclear
import histomicstk.utils as htk_utils
from histomicstk.cli import utils as cli_utils
from histomicstk.cli.utils import CLIArgumentParser

logging.basicConfig(level=logging.CRITICAL)


def check_args(args):

    if not os.path.isfile(args.inputImageFile):
        msg = 'Input image file does not exist.'
        raise OSError(msg)

    if len(args.reference_mu_lab) != 3:
        msg = 'Reference Mean LAB should be a 3 element vector.'
        raise ValueError(msg)

    if len(args.reference_std_lab) != 3:
        msg = 'Reference Stddev LAB should be a 3 element vector.'
        raise ValueError(msg)

    if len(args.analysis_roi) != 4:
        msg = 'Analysis ROI must be a vector of 4 elements.'
        raise ValueError(msg)

    if os.path.splitext(args.outputNucleiFeatureFile)[1] not in ['.csv', '.h5']:
        msg = 'Extension of output feature file must be .csv or .h5'
        raise ValueError(msg)


def main(args):  # noqa
    import dask
    import pandas as pd

    total_start_time = time.time()

    print('\n>> CLI Parameters ...\n')

    print(args)

    check_args(args)

    feature_file_format = os.path.splitext(args.outputNucleiFeatureFile)[1]

    if np.all(np.array(args.analysis_roi) == -1):
        process_whole_image = True
    else:
        process_whole_image = False

    # Provide default value for tile_overlap
    tile_overlap = args.tile_overlap_value
    if tile_overlap == -1:
        tile_overlap = (args.max_radius + 1) * 4

    #
    # Initiate Dask client
    #
    print('\n>> Creating Dask client ...\n')

    start_time = time.time()

    c = cli_utils.create_dask_client(args)

    print(c)

    dask_setup_time = time.time() - start_time
    print(f'Dask setup time = {dask_setup_time} seconds')

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
        'tile_overlap': {'x': tile_overlap, 'y': tile_overlap},
    }

    if not process_whole_image:

        it_kwargs['region'] = {
            'left': args.analysis_roi[0],
            'top': args.analysis_roi[1],
            'width': args.analysis_roi[2],
            'height': args.analysis_roi[3],
            'units': 'base_pixels',
        }

    if is_wsi:

        print('\n>> Computing foreground fraction of all tiles ...\n')

        start_time = time.time()

        num_tiles = ts.getSingleTile(**it_kwargs)['iterator_range']['position']

        print(f'Number of tiles = {num_tiles}')

        if process_whole_image:

            tile_fgnd_frac_list = htk_utils.compute_tile_foreground_fraction(
                args.inputImageFile, im_fgnd_mask_lres, fgnd_seg_scale,
                it_kwargs,
            )

        else:

            tile_fgnd_frac_list = np.full(num_tiles, 1.0)

        num_fgnd_tiles = np.count_nonzero(
            tile_fgnd_frac_list >= args.min_fgnd_frac)

        percent_fgnd_tiles = 100.0 * num_fgnd_tiles / num_tiles

        fgnd_frac_comp_time = time.time() - start_time

        print('Number of foreground tiles = {:d} ({:.2f}%)'.format(
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
    # Detect and compute nuclei features in parallel using Dask
    #
    print('\n>> Detecting nuclei and computing features ...\n')

    start_time = time.time()

    tile_result_list = []

    for tile in ts.tileIterator(**it_kwargs):

        tile_position = tile['tile_position']['position']

        if is_wsi and tile_fgnd_frac_list[tile_position] <= args.min_fgnd_frac:
            continue

        # detect nuclei
        cur_result = dask.delayed(htk_nuclear.detect_tile_nuclei)(
            tile,
            args,
            src_mu_lab, src_sigma_lab,
            return_fdata=True,
        )

        # append result to list
        tile_result_list.append(cur_result)

    tile_result_list = dask.delayed(tile_result_list).compute()

    nuclei_annot_list = [annot
                         for annot_list, fdata in tile_result_list
                         for annot in annot_list]

    nuclei_fdata = pd.DataFrame()

    if len(nuclei_annot_list) > 0:

        nuclei_fdata = pd.concat([
            fdata
            for annot_list, fdata in tile_result_list if fdata is not None],
            ignore_index=True,
        )

    nuclei_detection_time = time.time() - start_time

    print(f'Number of nuclei = {len(nuclei_annot_list)}')
    print('Nuclei detection time = {}'.format(
        cli_utils.disp_time_hms(nuclei_detection_time)))

    #
    # Remove overlapping nuclei
    #
    if args.remove_overlapping_nuclei_segmentation:
        print('\n>> Removing overlapping nuclei segmentations ...\n')
        nuclei_removal_start_time = time.time()

        nuclei_annot_list, selected_nuclei = htk_seg_label.remove_overlap_nuclei(
            nuclei_annot_list, args.nuclei_annotation_format, return_selected_nuclei=True)
        nuclei_removal_setup_time = time.time() - nuclei_removal_start_time

        # Remove features for overlapping nuclei
        nuclei_fdata = nuclei_fdata[nuclei_fdata.index.isin(selected_nuclei)]

        print(f'Number of nuclei after overlap removal {len(nuclei_annot_list)}')
        print('Nuclei removal processing time = {}'.format(
            cli_utils.disp_time_hms(nuclei_removal_setup_time)))

    #
    # Write annotation file
    #
    print('\n>> Writing annotation file ...\n')

    annot_fname = os.path.splitext(
        os.path.basename(args.outputNucleiAnnotationFile))[0]

    if args.in_annotations:
        for key in args.in_annotations.split(','):
            if key in nuclei_fdata:
                for row, value in zip(nuclei_annot_list, nuclei_fdata[key]):
                    if value is not None:
                        row.setdefault('user', {})[key] = value
    annotation = {
        'name': annot_fname + '-nuclei-' + args.nuclei_annotation_format,
        'elements': nuclei_annot_list,
        'attributes': {
            'src_mu_lab': None if src_mu_lab is None else src_mu_lab.tolist(),
            'src_sigma_lab': None if src_sigma_lab is None else src_sigma_lab.tolist(),
            'params': vars(args),
            'cli': Path(__file__).stem,
            'version': histomicstk.__version__},
    }

    with open(args.outputNucleiAnnotationFile, 'w') as annotation_file:
        json.dump(annotation, annotation_file, separators=(',', ':'), sort_keys=False)

    #
    # Create CSV Feature file
    #
    print('>> Writing CSV feature file')

    if feature_file_format == '.csv':

        nuclei_fdata.to_csv(args.outputNucleiFeatureFile, index=False)

    elif feature_file_format == '.h5':

        nuclei_fdata.to_hdf(args.outputNucleiFeatureFile, 'Features',
                            format='table', mode='w')

    else:

        msg = 'Extension of output feature file must be .csv or .h5'
        raise ValueError(msg)

    total_time_taken = time.time() - total_start_time

    print('Total analysis time = {}'.format(
        cli_utils.disp_time_hms(total_time_taken)))


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
