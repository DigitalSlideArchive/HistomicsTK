from ctk_cli import CLIArgumentParser
from dask import delayed
from dask.distributed import Client
import large_image
import numpy as np
import skimage.io


def main(args):
    Client(args.scheduler_address or None)
    ts = large_image.getTileSource(args.inputImageFile)
    kwargs = dict(format=large_image.tilesource.TILE_FORMAT_NUMPY)
    useWholeImage = args.region == [-1] * 4
    if not useWholeImage:
        kwargs['region'] = dict(zip(['left', 'top', 'width', 'height'],
                                    args.region))

    makeLabelImage = args.outputLabelImage is not None
    if makeLabelImage:
        if args.maxRegionSize != -1:
            if useWholeImage:
                size = max(ts.sizeX, ts.sizeY)
            else:
                size = max(args.region[-2:])
            if size > args.maxRegionSize:
                raise ValueError('Requested region is too large!  Please see --maxRegionSize')
        tile = ts.getRegion(**kwargs)[0]
        results, labelImage = positive_pixel_count_single_tile(args, tile, makeLabelImage=True)
        skimage.io.imsave(args.outputLabelImage, labelImage)
    else:
        results = []
        total_tiles = ts.getSingleTile(**kwargs)['iterator_range']['position']
        for position in range(0, total_tiles, args.tile_grouping):
            results.append(delayed(positive_pixel_count_tiles)(
                args, kwargs, position, min(args.tile_grouping, total_tiles - position)))
        results = delayed(combine)(results).compute()

    total_all_positive = results[:3].sum()
    output = (zip(['NumberWeakPositive', 'NumberPositive', 'NumberStrongPositive',
                   'IntensitySumWeakPositive', 'IntensitySumPositive', 'IntensitySumStrongPositive'],
                  results)
              + [('IntensityAverage', results[3:].sum() / total_all_positive),
                 ('RatioStrongToTotal', results[2] / total_all_positive),
                 ('IntensityAverageWeakAndPositive', results[3:5].sum() / results[:2].sum())])
    with open(args.returnParameterFile, 'w') as f:
        for k, v in output:
            f.write('{} = {}\n'.format(k, v))


def combine(results):
    return np.sum(results, 0) if results else np.zeros(6)


def positive_pixel_count_tiles(args, kwargs, position, count):
    ts = large_image.getTileSource(args.inputImageFile)
    total = np.zeros(6) # In order: #weak, #+ve, #strong, sum_weak, sum_+ve, sum_strong
    for pos in range(position, position + count):
        tile = ts.getSingleTile(tile_position=pos, **kwargs)['tile']
        total += positive_pixel_count_single_tile(args, tile, makeLabelImage=False)
    return total


def positive_pixel_count_single_tile(args, tile, makeLabelImage):
    tile = tile[..., :3]
    total = np.empty(6)
    tile_hsi = rgb_to_hsi(tile / 255.)
    mask_all_positive = (
        (np.abs((tile_hsi[..., 0] - args.hueValue + 0.5 % 1) - 0.5) <= args.hueWidth / 2.)
        & (tile_hsi[..., 1] >= args.saturationMinimum)
        & (tile_hsi[..., 2] < args.intensityUpperLimit)
        & (tile_hsi[..., 2] >= args.intensityLowerLimit)
    )
    all_positive_i = tile_hsi[mask_all_positive, 2]
    mask_weak = all_positive_i >= args.intensityWeakThreshold
    total[[0, 3]] = np.count_nonzero(mask_weak), np.sum(all_positive_i[mask_weak])
    mask_strong = all_positive_i < args.intensityStrongThreshold
    total[[2, 5]] = np.count_nonzero(mask_strong), np.sum(all_positive_i[mask_strong])
    mask_pos = ~(mask_weak | mask_strong)
    total[[1, 4]] = np.count_nonzero(mask_pos), np.sum(all_positive_i[mask_pos])
    if makeLabelImage:
        labelImage = np.full_like(tile, 255)
        # Colors from the "coolwarm" color map
        labelImage[mask_all_positive] = (
            mask_weak[..., np.newaxis] * [60, 78, 194]
            + mask_pos[..., np.newaxis] * [221, 220, 220]
            + mask_strong[..., np.newaxis] * [180, 4, 38]
        )
        return total, labelImage
    else:
        return total


def rgb_to_hsi(im):
    """Convert to HSI the RGB pixels in im.  Adapted from
    https://en.wikipedia.org/wiki/HSL_and_HSV#Hue_and_chroma."""
    im = np.moveaxis(im, -1, 0)
    hues = (np.arctan2(3**0.5 * (im[1] - im[2]),
                       2 * im[0] - im[1] - im[2]) / (2 * np.pi)) % 1
    intensities = im.mean(0)
    saturations = np.where(intensities, 1 - im.min(0) / intensities, 0)
    return np.stack([hues, saturations, intensities], -1)


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
