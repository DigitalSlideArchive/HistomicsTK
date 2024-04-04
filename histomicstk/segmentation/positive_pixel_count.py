from collections import namedtuple

import large_image
import numpy as np

from ..preprocessing.color_conversion import rgb_to_hsi


# This can be an enum in Python >= 3.4
class Labels:
    """Labels for the output image of the positive pixel count routines."""

    NEGATIVE = 0
    WEAK = 1
    PLAIN = 2
    STRONG = 3


class Parameters(
        namedtuple('Parameters', [
            'hue_value',
            'hue_width',
            'saturation_minimum',
            'intensity_upper_limit',
            'intensity_weak_threshold',
            'intensity_strong_threshold',
            'intensity_lower_limit',
        ]),
):
    """Parameters(hue_value, hue_width, saturation_minimum,
    intensity_upper_limit, intensity_weak_threshold,
    intensity_strong_threshold, intensity_lower_limit)

    Attributes
    ----------
    hue_value:
      Center of the hue range in HSI space for the positive color, in
      the range [0, 1]
    hue_width:
      Width of the hue range in HSI space
    saturation_minimum:
      Minimum saturation of positive pixels in HSI space, in the range
      [0, 1]
    intensity_upper_limit:
      Intensity threshold in HSI space above which a pixel is
      considered negative, in the range [0, 1]
    intensity_weak_threshold:
      Intensity threshold in HSI space that separates weak-positive
      pixels (above) from plain positive pixels (below)
    intensity_strong_threshold:
      Intensity threshold in HSI space that separates plain positive
      pixels (above) from strong positive pixels (below)
    intensity_lower_limit:
      Intensity threshold in HSI space below which a pixel is
      considered negative

    """


OutputTotals = namedtuple('OutputTotals', [
    'NumberWeakPositive',
    'NumberPositive',
    'NumberStrongPositive',
    'NumberTotalPixels',
    'IntensitySumWeakPositive',
    'IntensitySumPositive',
    'IntensitySumStrongPositive',
])

Output = namedtuple('Output', OutputTotals._fields + (
    'IntensityAverage',
    'RatioStrongToTotal',
    'IntensityAverageWeakAndPositive',
    'RatioStrongToPixels',
    'RatioWeakToPixels',
    'RatioTotalToPixels',
))


def count_slide(slide_path, params, region=None,
                tile_grouping=256, make_label_image=False):
    """Compute a count of positive pixels in the slide at slide_path.
    This routine can also create a label image.

    Parameters
    ----------
    slide_path : string (path)
        Path to the slide to analyze.
    params : Parameters
        An instance of Parameters, which see for further documentation
    region : dict, optional
        A valid region dict (per a large_image
        TileSource.tileIterator's region argument)
    tile_grouping : int
        The number of tiles to process as part of a single task
    make_label_image : bool, default=False
        Whether to make a label image.  See also "Notes"

    Returns
    -------
    stats : Output
        Various statistics on the input image.  See Output.
    label_image : array-like, only if make_label_image is set

    Notes
    -----
    The return value is either a single or a pair -- it is in either
    case a tuple.  Dask is used as configured to compute the
    statistics, but only if make_label_image is reset.  If
    make_label_image is set, everything is computed in a
    single-threaded manner.

    """
    from dask import delayed

    ts = large_image.getTileSource(slide_path)
    kwargs = dict(format=large_image.tilesource.TILE_FORMAT_NUMPY)
    if region is not None:
        kwargs['region'] = region
    if make_label_image:
        tile = ts.getRegion(**kwargs)[0]
        return count_image(tile, params)
    else:
        results = []
        total_tiles = ts.getSingleTile(**kwargs)['iterator_range']['position']
        for position in range(0, total_tiles, tile_grouping):
            results.append(delayed(_count_tiles)(
                slide_path, params, kwargs, position,
                min(tile_grouping, total_tiles - position)))
        results = delayed(_combine)(results).compute()
    return (_totals_to_stats(results),)


def _combine(results):
    return OutputTotals._make(sum(r[i] for r in results)
                              for i in range(len(OutputTotals._fields)))


def _count_tiles(slide_path, params, kwargs, position, count):
    ts = large_image.getTileSource(slide_path)
    lpotf = len(OutputTotals._fields)
    total = [0] * lpotf
    for pos in range(position, position + count):
        tile = ts.getSingleTile(tile_position=pos, **kwargs)['tile']
        subtotal = _count_image(tile, params)[0]
        for k in range(lpotf):
            total[k] += subtotal[k]
    return OutputTotals._make(total)


def count_image(image, params, mask=None):
    """Count positive pixels, computing a label mask and summary
    statistics.

    Parameters
    ----------
    image : array-like
        NxMx3 array of RGB data
    params : Parameters
        An instance of Parameters, which see for further documentation
    mask: array-like
        A boolean mask.  If present, only pixels where the mask is True are
        considered.

    Returns
    -------
    stats : Output
        Various statistics on the input image.  See Output.
    label_image : array-like
        NxM array of pixel types.  See Labels for the different values.

    """
    total, masks = _count_image(image, params, mask)
    mask_all_positive, mask_weak, mask_pos, mask_strong = masks
    label_image = np.full(image.shape[:-1], Labels.NEGATIVE, dtype=np.uint8)
    label_image[mask_all_positive] = (
        mask_weak * Labels.WEAK +
        mask_pos * Labels.PLAIN +
        mask_strong * Labels.STRONG
    )
    return _totals_to_stats(total), label_image


def _count_image(image, params, mask=None):
    """A version of count_image that doesn't compute the label image and
    only computes the sums.

    """
    p = params
    image_hsi = rgb_to_hsi(image / 255)
    mask_all_positive = (
        (np.abs(((image_hsi[..., 0] - p.hue_value + 0.5) % 1) - 0.5) <=
         p.hue_width / 2) &
        (image_hsi[..., 1] >= p.saturation_minimum) &
        (image_hsi[..., 2] < p.intensity_upper_limit) &
        (image_hsi[..., 2] >= p.intensity_lower_limit)
    )
    if mask is not None:
        mask_all_positive &= mask
    all_positive_i = image_hsi[mask_all_positive, 2]
    mask_weak = all_positive_i >= p.intensity_weak_threshold
    nw, iw = np.count_nonzero(mask_weak), np.sum(all_positive_i[mask_weak])
    mask_strong = all_positive_i < p.intensity_strong_threshold
    ns, is_ = np.count_nonzero(mask_strong), np.sum(all_positive_i[mask_strong])
    mask_pos = ~(mask_weak | mask_strong)
    np_, ip = np.count_nonzero(mask_pos), np.sum(all_positive_i[mask_pos])
    total = OutputTotals(
        NumberWeakPositive=nw,
        NumberPositive=np_,
        NumberStrongPositive=ns,
        NumberTotalPixels=(np.count_nonzero(mask) if mask is not None else
                           image_hsi.shape[0] * image_hsi.shape[1]),
        IntensitySumWeakPositive=iw,
        IntensitySumPositive=ip,
        IntensitySumStrongPositive=is_,
    )
    return total, (mask_all_positive, mask_weak, mask_pos, mask_strong)


def _totals_to_stats(total):
    """Do the extra computations to convert an OutputTotals to an Output"""
    t = total
    all_positive = t.NumberWeakPositive + t.NumberPositive + t.NumberStrongPositive
    return Output(
        IntensityAverage=((t.IntensitySumWeakPositive +
                           t.IntensitySumPositive +
                           t.IntensitySumStrongPositive) /
                          all_positive) if all_positive else 0,
        RatioStrongToTotal=t.NumberStrongPositive / all_positive if all_positive else 0,
        IntensityAverageWeakAndPositive=(
            (t.IntensitySumWeakPositive + t.IntensitySumPositive) /
            (t.NumberWeakPositive + t.NumberPositive)
        ) if t.NumberWeakPositive + t.NumberPositive else 0,
        RatioStrongToPixels=t.NumberStrongPositive / (t.NumberTotalPixels or 1),
        RatioWeakToPixels=t.NumberWeakPositive / (t.NumberTotalPixels or 1),
        RatioTotalToPixels=all_positive / (t.NumberTotalPixels or 1),
        **t._asdict(),
    )


__all__ = (
    'Parameters',
    'Output',
    'count_slide',
    'count_image',
)
