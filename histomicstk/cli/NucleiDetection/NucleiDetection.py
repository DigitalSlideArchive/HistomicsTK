import json
import logging
import os
import pprint
import time
from pathlib import Path

import large_image
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

import histomicstk
import histomicstk.preprocessing.color_deconvolution as htk_cdeconv
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
    if args.ImageInversionForm == "Yes":
        invert_image = True
    if args.ImageInversionForm == "No":
        invert_image = False
    if args.ImageInversionForm == "default":
        default_img_inversion = True
    return invert_image, default_img_inversion


def detect_tile_nuclei(tile_info, args, src_mu_lab=None,
                       src_sigma_lab=None, invert_image=False,
                       default_img_inversion=False):
    # Flags
    single_channel = False

    # get tile image & check number of channels
    single_channel = len(tile_info['tile'].shape) <= 2 or tile_info['tile'].shape[2] == 1
    if single_channel:
        im_tile = np.dstack((tile_info['tile'], tile_info['tile'], tile_info['tile']))
        if default_img_inversion:
            invert_image = True
    else:
        im_tile = tile_info['tile'][:, :, :3]

    # perform image inversion
    if invert_image:
        im_tile = np.max(im_tile) - im_tile

    # perform color normalization
    im_nmzd = htk_cnorm.reinhard(im_tile,
                                 args.reference_mu_lab,
                                 args.reference_std_lab,
                                 src_mu=src_mu_lab,
                                 src_sigma=src_sigma_lab)

    # perform color decovolution
    w = cli_utils.get_stain_matrix(args)

    # perform deconvolution
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

    # Delete overlapping border nuclei
    if any(tile_info['tile_overlap'].values()) > 0:

        im_nuclei_seg_mask = htk_seg_label.delete_overlap(
            im_nuclei_seg_mask, overlap_info=tile_info['tile_overlap'])

    # generate nuclei annotations
    nuclei_annot_list = []

    flag_nuclei_found = np.any(im_nuclei_seg_mask)

    if flag_nuclei_found:
        nuclei_annot_list = cli_utils.create_tile_nuclei_annotations(
            im_nuclei_seg_mask, tile_info, args.nuclei_annotation_format)

    return nuclei_annot_list


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
            it_kwargs, style=args.style
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

    print('Number of foreground tiles = {:d} ({:2f}%%)'.format(
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
        cur_nuclei_list = dask.delayed(detect_tile_nuclei)(
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


def overlapping_nuclei_removal(nuclei_list, search_area):
    print('\n>> Overlapping nuclei removal...\n')
    start_time = time.time()

    pop_list = []
    merged_polygons = []

    # Take the first element of the data list
    sample_element = nuclei_list[0]

    # Iterate over the elements in the data list along with their index
    for i, sample in enumerate(nuclei_list):
        # Find the maximum x-coordinate and y-coordinate in the current 'points' attribute
        curr_x = max([sublist[0] for sublist in sample['points']])
        curr_y = max([sublist[1] for sublist in sample['points']])

        # Iterate over the remaining elements
        for j in range(i + 1, len(nuclei_list) - 1):
            # Find the maximum x-coordinate and y-coordinate j-th element
            given_x = max([sublist[0] for sublist in nuclei_list[j]['points']])
            given_y = max([sublist[1] for sublist in nuclei_list[j]['points']])

            # Check if the maximum x and y coordinates of the j-th element are within given units
            if given_x < curr_x + search_area and given_y < curr_y + search_area:

                # Check if the polygons represented by the i-th and j-th elements intersect
                poly1 = make_valid(Polygon(sample['points']))
                poly2 = make_valid(Polygon(nuclei_list[j]['points']))
                if poly1.intersects(poly2):
                    # Merge the polygons using unary_union
                    merged_polygon = unary_union([poly1, poly2])

                    # Append the merged_polygon to the list of merged polygons
                    merged_polygons.append(merged_polygon)

                    # Add the current i and j indices to the pop_list if they are not already
                    # present
                    if i not in pop_list:
                        pop_list.append(i)
                    if j not in pop_list:
                        pop_list.append(j)

    def _remove_indexes(lst, indexes):
        # Sort the indexes in descending order
        indexes.sort(reverse=True)

        # Remove the elements at the specified indexes
        for index in indexes:
            if 0 <= index < len(lst):
                del lst[index]

        return lst

    nuclei_list = _remove_indexes(nuclei_list, pop_list)

    # convert all the data in merged_polygons to list
    overlapping_nuclei_count = 0
    if merged_polygons:
        for element in merged_polygons:

            if element.geom_type == 'GeometryCollection':
                for i in range(len(element.geoms)):
                    if element.geoms[i].geom_type == 'Polygon':
                        rlist = list(element.geoms[i].exterior.coords)

            elif element.geom_type == 'MultiPolygon':
                for polygon in list(element.geoms):
                    rlist = list(polygon.exterior.coords)

            else:
                rlist = list(element.exterior.coords)

            sample_element['points'] = rlist
            nuclei_list.append(sample_element)
            overlapping_nuclei_count += 1

    overlap_detection_time = time.time() - start_time
    print('Number of overlapping nuclei {}'.format(overlapping_nuclei_count))
    print('Number of nuclei after overlap removal {}'.format(len(nuclei_list)))
    print('Nuclei overlap detection time = {}\n'.format(
        cli_utils.disp_time_hms(overlap_detection_time)))

    return nuclei_list


def main(args):

    # Flags
    invert_image = False
    default_img_inversion = False
    process_whole_image = False

    total_start_time = time.time()

    print('\n>> CLI Parameters ...\n')
    pprint.pprint(vars(args))

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

    # retrive style
    if not args.style or args.style.startswith('{#control'):
        args.style = None

    # initial arguments
    it_kwargs = {
        'tile_size': {'width': args.analysis_tile_size},
        'scale': {'magnification': args.analysis_mag},
        'tile_overlap': {'x': args.tile_overlap_value, 'y': args.tile_overlap_value},
        'style': {args.style}
    }

    # retrive frame
    if not args.frame or args.frame.startswith('{#control'):
        args.frame = None
    elif not args.frame.isdigit():
        raise Exception("The given frame value is not an integer")
    else:
        it_kwargs['frame'] = args.frame

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
            'units': 'base_pixels'
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

    if args.merge_overlapping_nuclei_segmentation:
        nuclei_list = overlapping_nuclei_removal(nuclei_list, search_area=args.search_area_size)

    with open(args.outputNucleiAnnotationFile, 'w') as annotation_file:
        json.dump(annotation, annotation_file, separators=(',', ':'), sort_keys=False)

    total_time_taken = time.time() - total_start_time

    print('Total analysis time = {}'.format(
        cli_utils.disp_time_hms(total_time_taken)))


if __name__ == '__main__':

    main(CLIArgumentParser().parse_args())
