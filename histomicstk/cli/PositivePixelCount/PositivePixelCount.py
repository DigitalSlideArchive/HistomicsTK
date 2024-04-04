import json
import pprint
import time
from pathlib import Path

import large_image
import numpy as np

import histomicstk
import histomicstk.segmentation.positive_pixel_count as ppc
from histomicstk.cli import utils
from histomicstk.cli.utils import CLIArgumentParser


def tile_positive_pixel_count(imagePath, tilePosition, it_kwargs, ppc_params,
                              color_map, useAlpha, region_polygons, style):
    # Start measuring time
    tile_start_time = time.time()

    # Get the tile source
    ts = large_image.getTileSource(imagePath, style=style)

    # Fetch the image tile
    tile = ts.getSingleTile(tile_position=tilePosition, **it_kwargs)

    # Convert region polygons to binary mask
    mask = utils.polygons_to_binary_mask(
        region_polygons, tile['x'], tile['y'], tile['width'], tile['height'])

    # Get the tile image
    img = tile['tile']

    # Handle single-channel images (convert to 3 channels)
    if len(img.shape) == 2:
        img = img.reshape((img.shape[0], img.shape[1], 1))
    if len(img.shape) == 3 and img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=2)

    # Calculate the positive pixel count and generate a mask
    result, ppcmask = ppc.count_image(img, ppc_params, mask)

    # Release the tile to free up resources
    tile.release()

    # Apply the color map to the mask to generate an image with colored regions
    ppcimg = color_map[ppcmask]

    # Optionally remove alpha channel if not used
    if not useAlpha:
        ppcimg = ppcimg[:, :, :3]

    # Return the result, colored image, tile's x and y coordinates, mask, and
    # tile processing start time
    return result, ppcimg, tile['x'], tile['y'], mask, tile_start_time


def main(opts):
    # Print the options dictionary using pprint
    pprint.pprint(vars(opts))

    # Handle style option
    if not opts.style or opts.style.startswith('{#control'):
        opts.style = None

    # Get the tile source
    ts = large_image.getTileSource(opts.inputImageFile, style=opts.style)

    # Create a sink for the output label image if specified
    sink = large_image.new() if getattr(opts, 'outputLabelImage', None) else None

    # Get region parameters and polygons
    tiparams = utils.get_region_dict(opts.region, None, ts)
    region_polygons = utils.get_region_polygons(opts.region)

    # Print region parameters and polygons
    print('region: %r %r' % (tiparams, region_polygons))

    # Set tile size and useAlpha flag
    tileSize = 4096
    useAlpha = len(opts.region) > 6

    # Define the color map for label images
    color_map = np.empty((4, 4), dtype=np.uint8)
    color_map[ppc.Labels.NEGATIVE] = 255, 255, 255, 255
    color_map[ppc.Labels.WEAK] = 60, 78, 194, 255
    color_map[ppc.Labels.PLAIN] = 221, 220, 220, 255
    color_map[ppc.Labels.STRONG] = 180, 4, 38, 255

    # Set positive pixel count parameters
    ppc_params = ppc.Parameters(
        **{k: getattr(opts, k) for k in ppc.Parameters._fields},
    )

    # Initialize a list to store results
    results = []

    # Set crop coordinates if 'left' is present in the region parameters
    if sink and 'left' in tiparams.get('region', {}):
        sink.crop = (
            tiparams['region']['left'], tiparams['region']['top'],
            tiparams['region']['width'], tiparams['region']['height'])

    # Set parameters for tile source fetching
    tiparams['format'] = large_image.constants.TILE_FORMAT_NUMPY
    tiparams['tile_size'] = dict(width=tileSize, height=tileSize)
    try:
        tiparams['frame'] = int(opts.frame)
    except Exception:
        pass

    # Get the total number of tiles
    tileCount = next(ts.tileIterator(**tiparams))['iterator_range']['position']

    # Start processing tiles
    start_time = time.time()

    if tileCount > 4 and getattr(opts, 'scheduler', None) != 'none':
        # If tile count is large and Dask scheduler is not disabled, use Dask for
        # parallel processing
        print('>> Creating Dask client')
        client = utils.create_dask_client(opts)
        dask_setup_time = time.time() - start_time
        print(f'Dask setup time = {utils.disp_time_hms(dask_setup_time)}')
        futureList = []
        for tile in ts.tileIterator(**tiparams):
            tile_position = tile['tile_position']['position']
            # Submit the tile processing function to Dask client
            futureList.append(client.submit(
                tile_positive_pixel_count,
                opts.inputImageFile, tile_position, tiparams, ppc_params,
                color_map, useAlpha, region_polygons, opts.style))
        for idx, future in enumerate(futureList):
            # Wait for the results from Dask futures and process each tile
            result, ppcimg, x, y, mask, tile_start_time = future.result()
            results.append(result)
            if sink:
                sink.addTile(ppcimg, x, y, mask=mask)
            print('Processed tile %d/%d\n  %r\n  time %s (%s from start)' % (
                idx, tileCount, result,
                utils.disp_time_hms(time.time() - tile_start_time),
                utils.disp_time_hms(time.time() - start_time)))
    else:
        # If tile count is small or Dask scheduler is disabled, process tiles sequentially
        for tile in ts.tileIterator(**tiparams):
            tile_position = tile['tile_position']['position']
            result, ppcimg, x, y, mask, tile_start_time = tile_positive_pixel_count(
                opts.inputImageFile, tile_position, tiparams, ppc_params,
                color_map, useAlpha, region_polygons, opts.style)
            results.append(result)
            if sink:
                sink.addTile(ppcimg, x, y, mask=mask)
            print('Processed tile %d/%d\n  %r\n  time %s (%s from start)' % (
                tile_position, tileCount, result,
                utils.disp_time_hms(time.time() - tile_start_time),
                utils.disp_time_hms(time.time() - start_time)))

    print('Combining results, time from start %s' % (utils.disp_time_hms(time.time() - start_time)))

    # Combine the results and calculate statistics
    stats = ppc._totals_to_stats(ppc._combine(results))

    if sink:
        # Output the result to the label image file if specified
        print('Outputting file, time from start %s' % (
            utils.disp_time_hms(time.time() - start_time)))
        sink_start_time = time.time()
        sink.write(opts.outputLabelImage, lossy=False)
        print('Output file, time %s (%s from start)' % (
            utils.disp_time_hms(time.time() - sink_start_time),
            utils.disp_time_hms(time.time() - start_time)))

    # Print the calculated statistics
    pprint.pprint(stats._asdict())

    # Create an annotation dictionary
    annotation = {
        'name': 'Positive Pixel Count',
        'elements': [{
            'type': 'image',
            'girderId': 'outputLabelImage',
            'hasAlpha': useAlpha,
            'transform': {
                'xoffset': tiparams.get('region', {}).get('left', 0),
                'yoffset': tiparams.get('region', {}).get('top', 0),
            },
        }],
        'attributes': {
            'params': vars(opts),
            'stats': stats._asdict(),
            'cli': Path(__file__).stem,
            'version': histomicstk.__version__,
        },
    }

    # Save the annotation dictionary to the output annotation file
    if opts.outputAnnotationFile:
        with open(opts.outputAnnotationFile, 'w') as annotation_file:
            json.dump(annotation, annotation_file, separators=(',', ':'), sort_keys=False)
        print('Finished time %s' % (utils.disp_time_hms(time.time() - start_time)))


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
