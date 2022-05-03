import json
import pprint
import time

import large_image
import numpy as np

import histomicstk.segmentation.positive_pixel_count as ppc
from histomicstk.cli import utils
from histomicstk.cli.utils import CLIArgumentParser


def tile_positive_pixel_count(imagePath, tilePosition, it_kwargs, ppc_params,
                              color_map, useAlpha, region_polygons):
    tile_start_time = time.time()
    ts = large_image.getTileSource(imagePath)
    tile = ts.getSingleTile(tile_position=tilePosition, **it_kwargs)
    mask = utils.polygons_to_binary_mask(
        region_polygons, tile['x'], tile['y'], tile['width'], tile['height'])
    result, ppcmask = ppc.count_image(tile['tile'], ppc_params, mask)
    tile.release()
    ppcimg = color_map[ppcmask]
    if not useAlpha:
        ppcimg = ppcimg[:, :, :3]
    return result, ppcimg, tile['x'], tile['y'], mask, tile_start_time


def main(opts):
    pprint.pprint(vars(opts))
    ts = large_image.getTileSource(opts.inputImageFile)
    sink = large_image.new() if getattr(opts, 'outputLabelImage', None) else None
    tiparams = utils.get_region_dict(opts.region, None, ts)
    region_polygons = utils.get_region_polygons(opts.region)
    print('region: %r %r' % (tiparams, region_polygons))
    tileSize = 2048
    useAlpha = len(opts.region) > 6
    # Colorize label image.  Colors from the "coolwarm" color map
    color_map = np.empty((4, 4), dtype=np.uint8)
    color_map[ppc.Labels.NEGATIVE] = 255, 255, 255, 255
    color_map[ppc.Labels.WEAK] = 60, 78, 194, 255
    color_map[ppc.Labels.PLAIN] = 221, 220, 220, 255
    color_map[ppc.Labels.STRONG] = 180, 4, 38, 255
    ppc_params = ppc.Parameters(
        **{k: getattr(opts, k) for k in ppc.Parameters._fields}
    )
    results = []
    if 'left' in tiparams.get('region', {}):
        sink.crop = (
            tiparams['region']['left'], tiparams['region']['top'],
            tiparams['region']['width'], tiparams['region']['height'])
    tiparams['format'] = large_image.constants.TILE_FORMAT_NUMPY
    tiparams['tile_size'] = dict(width=tileSize, height=tileSize)
    tileCount = next(ts.tileIterator(**tiparams))['iterator_range']['position']
    start_time = time.time()
    if tileCount > 4 and getattr(opts, 'scheduler', None) != 'none':
        print('>> Creating Dask client')

        client = utils.create_dask_client(opts)
        dask_setup_time = time.time() - start_time
        print('Dask setup time = {}'.format(utils.disp_time_hms(dask_setup_time)))
        futureList = []
        for tile in ts.tileIterator(**tiparams):
            tile_position = tile['tile_position']['position']
            futureList.append(client.submit(
                tile_positive_pixel_count,
                opts.inputImageFile, tile_position, tiparams, ppc_params,
                color_map, useAlpha, region_polygons))
        for idx, future in enumerate(futureList):
            result, ppcimg, x, y, mask, tile_start_time = future.result()
            results.append(result)
            if sink:
                sink.addTile(ppcimg, x, y, mask=mask)
            print('Processed tile %d/%d\n  %r\n  time %s (%s from start)' % (
                idx, tileCount, result,
                utils.disp_time_hms(time.time() - tile_start_time),
                utils.disp_time_hms(time.time() - start_time)))
    else:
        for tile in ts.tileIterator(**tiparams):
            tile_position = tile['tile_position']['position']
            result, ppcimg, x, y, mask, tile_start_time = tile_positive_pixel_count(
                opts.inputImageFile, tile_position, tiparams, ppc_params,
                color_map, useAlpha, region_polygons)
            results.append(result)
            if sink:
                sink.addTile(ppcimg, x, y, mask=mask)
            print('Processed tile %d/%d\n  %r\n  time %s (%s from start)' % (
                tile_position, tileCount, result,
                utils.disp_time_hms(time.time() - tile_start_time),
                utils.disp_time_hms(time.time() - start_time)))
    print('Combining results, time from start %s' % (utils.disp_time_hms(time.time() - start_time)))
    stats = ppc._totals_to_stats(ppc._combine(results))
    if sink:
        print('Outputting file, time from start %s' % (
            utils.disp_time_hms(time.time() - start_time)))
        sink_start_time = time.time()
        sink.write(opts.outputLabelImage, lossy=False)
        print('Output file, time %s (%s from start)' % (
            utils.disp_time_hms(time.time() - sink_start_time),
            utils.disp_time_hms(time.time() - start_time)))
    pprint.pprint(stats._asdict())
    with open(opts.returnParameterFile, 'w') as f:
        for k, v in zip(stats._fields, stats):
            f.write(f'{k} = {v}\n')
    annotation = {
        'name': 'Positive Pixel Count',
        'description': 'Used params: %r\nResults: %r' % (vars(opts), stats._asdict()),
        'elements': [{
            'type': 'image',
            'girderId': 'outputLabelImage',
            'hasAlpha': useAlpha,
            'transform': {
                'xoffset': tiparams.get('region', {}).get('left', 0),
                'yoffset': tiparams.get('region', {}).get('top', 0),
            },
        }],
    }
    with open(opts.outputAnnotationFile, 'w') as annotation_file:
        json.dump(annotation, annotation_file, indent=2, sort_keys=False)
    print('Finished time %s' % (utils.disp_time_hms(time.time() - start_time)))


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
