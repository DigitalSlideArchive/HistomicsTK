import os
import sys

from ctk_cli import CLIArgumentParser
from dask import delayed
from dask.distributed import Client
import large_image
import numpy as np
import skimage.io

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), '..')))
from cli_common import utils  # noqa

results_num_keys = ('NumberWeakPositive', 'NumberPositive',
                    'NumberStrongPositive')
results_i_keys = ('IntensitySumWeakPositive', 'IntensitySumPositive',
                  'IntensitySumStrongPositive')
results_keys = results_num_keys + results_i_keys


def main(args):
    Client(args.scheduler_address or None)
    ts = large_image.getTileSource(args.inputImageFile)
    kwargs = dict(format=large_image.tilesource.TILE_FORMAT_NUMPY)
    makeLabelImage = args.outputLabelImage is not None
    kwargs.update(utils.get_region_dict(
        args.region,
        *(args.maxRegionSize, ts) if makeLabelImage else ()
    ))
    if makeLabelImage:
        tile = ts.getRegion(**kwargs)[0]
        results, labelImage = positive_pixel_count_single_tile(
            args, tile, makeLabelImage=True)
        skimage.io.imsave(args.outputLabelImage, labelImage)
    else:
        results = []
        total_tiles = ts.getSingleTile(**kwargs)['iterator_range']['position']
        for position in range(0, total_tiles, args.tile_grouping):
            results.append(delayed(positive_pixel_count_tiles)(
                args, kwargs, position,
                min(args.tile_grouping, total_tiles - position)))
        results = delayed(combine)(results).compute()

    r = results
    total_all_positive = sum(r[k] for k in results_num_keys)
    output = (
        [(k, r[k]) for k in results_keys] +
        [('IntensityAverage',
          sum(r[k] for k in results_i_keys) / total_all_positive),
         ('RatioStrongToTotal',
          float(r['NumberStrongPositive']) / total_all_positive),
         ('IntensityAverageWeakAndPositive',
          (r['IntensitySumWeakPositive'] + r['IntensitySumPositive']) /
          (r['NumberWeakPositive'] + r['NumberPositive']))]
    )
    with open(args.returnParameterFile, 'w') as f:
        for k, v in output:
            f.write('{} = {}\n'.format(k, v))


def combine(results):
    return {k: sum(r[k] for r in results) for k in results_keys}


def positive_pixel_count_tiles(args, kwargs, position, count):
    ts = large_image.getTileSource(args.inputImageFile)
    total = dict((k, 0) for k in results_keys)
    for pos in range(position, position + count):
        tile = ts.getSingleTile(tile_position=pos, **kwargs)['tile']
        subtotal = positive_pixel_count_single_tile(args, tile, makeLabelImage=False)
        for k in results_keys:
            total[k] += subtotal[k]
    return total


def positive_pixel_count_single_tile(args, tile, makeLabelImage):
    tile = tile[..., :3]
    tile_hsi = rgb_to_hsi(tile / 255.)
    mask_all_positive = (
        (np.abs((tile_hsi[..., 0] - args.hueValue + 0.5 % 1) - 0.5) <=
         args.hueWidth / 2.) &
        (tile_hsi[..., 1] >= args.saturationMinimum) &
        (tile_hsi[..., 2] < args.intensityUpperLimit) &
        (tile_hsi[..., 2] >= args.intensityLowerLimit)
    )
    all_positive_i = tile_hsi[mask_all_positive, 2]
    mask_weak = all_positive_i >= args.intensityWeakThreshold
    nw, iw = np.count_nonzero(mask_weak), np.sum(all_positive_i[mask_weak])
    mask_strong = all_positive_i < args.intensityStrongThreshold
    ns, is_ = np.count_nonzero(mask_strong), np.sum(all_positive_i[mask_strong])
    mask_pos = ~(mask_weak | mask_strong)
    np_, ip = np.count_nonzero(mask_pos), np.sum(all_positive_i[mask_pos])
    total = dict(
        NumberWeakPositive=nw,
        NumberPositive=np_,
        NumberStrongPositive=ns,
        IntensitySumWeakPositive=iw,
        IntensitySumPositive=ip,
        IntensitySumStrongPositive=is_,
    )
    if makeLabelImage:
        labelImage = np.full_like(tile, 255)
        # Colors from the "coolwarm" color map
        labelImage[mask_all_positive] = (
            mask_weak[..., np.newaxis] * [60, 78, 194] +
            mask_pos[..., np.newaxis] * [221, 220, 220] +
            mask_strong[..., np.newaxis] * [180, 4, 38]
        )
        return total, labelImage
    else:
        return total


def rgb_to_hsi(im):
    """Convert to HSI the RGB pixels in im.  Adapted from
    https://en.wikipedia.org/wiki/HSL_and_HSV#Hue_and_chroma.

    """
    im = np.moveaxis(im, -1, 0)
    hues = (np.arctan2(3**0.5 * (im[1] - im[2]),
                       2 * im[0] - im[1] - im[2]) / (2 * np.pi)) % 1
    intensities = im.mean(0)
    saturations = np.where(intensities, 1 - im.min(0) / intensities, 0)
    return np.stack([hues, saturations, intensities], -1)


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
