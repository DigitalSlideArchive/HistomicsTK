import json
import logging
import os
import pprint
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


def read_input_image(args, process_whole_image=False):
    # read input image and check if it is WSI
    print('\n>> Reading input image ... \n')

    ts = large_image.getTileSource(args.inputImageFile, style=args.style)

    ts_metadata = ts.getMetadata()

    print(json.dumps(ts_metadata, indent=2))

    is_wsi = ts_metadata['magnification'] is not None

    return ts, is_wsi


def image_inversion_flag_setter(args=None):
    # generates image inversion flags
    invert_image, default_img_inversion = False, False
    if args.ImageInversionForm == 'Yes':
        invert_image = True
    if args.ImageInversionForm == 'No':
        invert_image = False
    if args.ImageInversionForm == 'default':
        default_img_inversion = True
    return invert_image, default_img_inversion


def validate_args(args):
    # validates the input arguments
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


def process_wsi_as_whole_image(ts, invert_image=False, args=None, default_img_inversion=False):
    print('\n>> Computing tissue/foreground mask at low-res ...\n')

    start_time = time.time()
    # segment wsi foreground at low resolution
    im_fgnd_mask_lres, fgnd_seg_scale = \
        cli_utils.segment_wsi_foreground_at_low_res(
            ts, invert_image=invert_image, frame=args.frame,
            default_img_inversion=default_img_inversion)

    fgnd_time = time.time() - start_time

    print('low-res foreground mask computation time = {}'.format(
        cli_utils.disp_time_hms(fgnd_time)))

    return im_fgnd_mask_lres, fgnd_seg_scale


def process_wsi(ts, it_kwargs, args, im_fgnd_mask_lres=None,
                fgnd_seg_scale=None, process_whole_image=False):

    # process the wsi
    print('\n>> Computing foreground fraction of all tiles ...\n')

    start_time = time.time()

    num_tiles = ts.getSingleTile(**it_kwargs)['iterator_range']['position']

    print(f'Number of tiles = {num_tiles}')

    if process_whole_image:

        tile_fgnd_frac_list = htk_utils.compute_tile_foreground_fraction(
            args.inputImageFile, im_fgnd_mask_lres, fgnd_seg_scale,
            it_kwargs, style=args.style,
        )

    else:

        tile_fgnd_frac_list = np.full(num_tiles, 1.0)

    num_fgnd_tiles = np.count_nonzero(
        tile_fgnd_frac_list >= args.min_fgnd_frac)

    if not num_fgnd_tiles:
        tile_fgnd_frac_list = np.full(num_tiles, 1.0)
        percent_fgnd_tiles = 100
        num_fgnd_tiles = np.count_nonzero(tile_fgnd_frac_list)
    else:
        percent_fgnd_tiles = 100.0 * num_fgnd_tiles / num_tiles

    fgnd_frac_comp_time = time.time() - start_time

    print('Number of foreground tiles = {:d} ({:.2f}%)'.format(
        num_fgnd_tiles, percent_fgnd_tiles))

    print('Tile foreground fraction computation time = {}'.format(
        cli_utils.disp_time_hms(fgnd_frac_comp_time)))

    return tile_fgnd_frac_list


def compute_reinhard_norm(args, invert_image=False, default_img_inversion=False):
    print('\n>> Computing reinhard color normalization stats ...\n')

    start_time = time.time()
    src_mu_lab, src_sigma_lab = htk_cnorm.reinhard_stats(
        args.inputImageFile, 0.01, magnification=args.analysis_mag,
        invert_image=invert_image, style=args.style, frame=args.frame,
        default_img_inversion=default_img_inversion)

    rstats_time = time.time() - start_time

    print('Reinhard stats computation time = {}'.format(
        cli_utils.disp_time_hms(rstats_time)))
    return src_mu_lab, src_sigma_lab


def detect_nuclei_with_dask(ts, tile_fgnd_frac_list, it_kwargs, args,
                            invert_image=False, is_wsi=False, src_mu_lab=None,
                            src_sigma_lab=None, default_img_inversion=False):

    import dask

    print('\n>> Detecting nuclei ...\n')

    start_time = time.time()

    tile_nuclei_list = []

    for tile in ts.tileIterator(**it_kwargs):

        tile_position = tile['tile_position']['position']

        if is_wsi and tile_fgnd_frac_list[tile_position] <= args.min_fgnd_frac:
            continue

        # detect nuclei
        cur_nuclei_list = dask.delayed(htk_nuclear.detect_tile_nuclei)(
            tile,
            args,
            src_mu_lab, src_sigma_lab, invert_image=invert_image,
            default_img_inversion=default_img_inversion,
        )

        # append result to list
        tile_nuclei_list.append(cur_nuclei_list)

    tile_nuclei_list = dask.delayed(tile_nuclei_list).compute()

    nuclei_list = [anot
                   for anot_list in tile_nuclei_list for anot in anot_list]

    nuclei_detection_time = time.time() - start_time

    print(f'Number of nuclei = {len(nuclei_list)}')

    print('Nuclei detection time = {}'.format(
        cli_utils.disp_time_hms(nuclei_detection_time)))
    return nuclei_list


def main(args):  # noqa

    # Flags
    invert_image = False
    default_img_inversion = False
    process_whole_image = False

    total_start_time = time.time()

    print('\n>> CLI Parameters ...\n')
    pprint.pprint(vars(args))

    validate_args(args)

    if np.all(np.array(args.analysis_roi) == -1):
        process_whole_image = True
    else:
        process_whole_image = False

    # Provide default value for tile_overlap
    tile_overlap = args.tile_overlap_value
    if tile_overlap == -1:
        tile_overlap = (args.max_radius + 1) * 4

    # retrieve style
    if not args.style or args.style.startswith('{#control'):
        args.style = None

    # initial arguments
    it_kwargs = {
        'tile_size': {'width': args.analysis_tile_size},
        'scale': {'magnification': args.analysis_mag},
        'tile_overlap': {'x': tile_overlap, 'y': tile_overlap},
        'style': args.style,
    }

    # retrieve frame
    if not args.frame or args.frame.startswith('{#control'):
        args.frame = None
    elif not args.frame.isdigit():
        msg = 'The given frame value is not an integer'
        raise Exception(msg)
    else:
        it_kwargs['frame'] = int(args.frame)

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
    # color inversion flag
    #
    invert_image, default_img_inversion = image_inversion_flag_setter(args)

    #
    # Read Input Image
    #
    ts, is_wsi = read_input_image(args, process_whole_image)

    #
    # Compute foreground fraction of tiles in parallel using Dask
    #
    tile_fgnd_frac_list = [1.0]

    if not process_whole_image:

        it_kwargs['region'] = {
            'left': args.analysis_roi[0],
            'top': args.analysis_roi[1],
            'width': args.analysis_roi[2],
            'height': args.analysis_roi[3],
            'units': 'base_pixels',
        }

    if is_wsi:

        if process_whole_image:

            im_fgnd_mask_lres, fgnd_seg_scale = process_wsi_as_whole_image(
                ts, invert_image=invert_image, args=args,
                default_img_inversion=default_img_inversion)
            tile_fgnd_frac_list = process_wsi(ts,
                                              it_kwargs,
                                              args,
                                              im_fgnd_mask_lres,
                                              fgnd_seg_scale,
                                              process_whole_image)
        else:
            tile_fgnd_frac_list = process_wsi(ts, it_kwargs, args)

    #
    # Compute reinhard stats for color normalization
    #
    src_mu_lab = None
    src_sigma_lab = None

    if is_wsi and process_whole_image:
        # get a tile
        tile_info = ts.getSingleTile(
            format=large_image.tilesource.TILE_FORMAT_NUMPY,
            frame=args.frame)
        # get tile image & check number of channels
        single_channel = len(tile_info['tile'].shape) <= 2 or tile_info['tile'].shape[2] == 1
        if not single_channel:
            src_mu_lab, src_sigma_lab = compute_reinhard_norm(
                args, invert_image=invert_image, default_img_inversion=default_img_inversion)

    if src_mu_lab is None:
        smallImage, _ = ts.getRegion(
            output=dict(maxWidth=4096, maxHeight=4096), resample=False,
            region=it_kwargs.get('region', {}),
            format=large_image.tilesource.TILE_FORMAT_NUMPY,
            frame=args.frame)
        if len(smallImage.shape) == 2:
            smallImage = np.resize(smallImage, (smallImage.shape[0], smallImage.shape[1], 1))
        if smallImage.shape[2] < 3:
            if default_img_inversion or invert_image:
                smallImage = (np.iinfo(smallImage.dtype).max
                              if smallImage.dtype.kind == 'u'
                              else np.max(smallImage)) - smallImage
            smallImage = np.repeat(smallImage[:, :, :1], 3, 2)
        elif not default_img_inversion and invert_image:
            smallImage = (np.iinfo(smallImage.dtype).max
                          if smallImage.dtype.kind == 'u'
                          else np.max(smallImage)) - smallImage
        smallImage = smallImage[:, :, :3]
        src_mu_lab, src_sigma_lab = htk_cnorm.reinhard_stats_rgb(smallImage)

    #
    # Detect nuclei in parallel using Dask
    #
    nuclei_list = detect_nuclei_with_dask(
        ts,
        tile_fgnd_frac_list,
        it_kwargs,
        args,
        invert_image,
        is_wsi,
        src_mu_lab,
        src_sigma_lab,
        default_img_inversion=default_img_inversion)

    #
    # Remove overlapping nuclei
    #
    if args.remove_overlapping_nuclei_segmentation:
        print('\n>> Removing overlapping nuclei segmentations ...\n')
        nuclei_removal_start_time = time.time()

        nuclei_list = htk_seg_label.remove_overlap_nuclei(
            nuclei_list, args.nuclei_annotation_format)
        nuclei_removal_setup_time = time.time() - nuclei_removal_start_time

        print(f'Number of nuclei after overlap removal {len(nuclei_list)}')
        print('Nuclei removal processing time = {}'.format(
            cli_utils.disp_time_hms(nuclei_removal_setup_time)))

    #
    # Write annotation file
    #
    print('\n>> Writing annotation file ...\n')

    annot_fname = os.path.splitext(
        os.path.basename(args.outputNucleiAnnotationFile))[0]

    annotation = {
        'name': annot_fname + '-nuclei-' + args.nuclei_annotation_format,
        'elements': nuclei_list,
        'attributes': {
            'src_mu_lab': None if src_mu_lab is None else src_mu_lab.tolist(),
            'src_sigma_lab': None if src_sigma_lab is None else src_sigma_lab.tolist(),
            'params': vars(args),
            'cli': Path(__file__).stem,
            'version': histomicstk.__version__,
        },
    }

    with open(args.outputNucleiAnnotationFile, 'w') as annotation_file:
        json.dump(annotation, annotation_file, separators=(',', ':'), sort_keys=False)

    total_time_taken = time.time() - total_start_time

    print('Total analysis time = {}'.format(
        cli_utils.disp_time_hms(total_time_taken)))


if __name__ == '__main__':

    main(CLIArgumentParser().parse_args())
