from __future__ import division

from collections import namedtuple

from dask import delayed
import large_image
import numpy as np

from .color_conversion import rgb_to_hsi


# This can be an enum in Python >= 3.4
class Labels(object):
    """Labels for the output image of the positive pixel count routines."""
    NEGATIVE = 0
    WEAK = 1
    PLAIN = 2
    STRONG = 3


class PPCParameters(
        namedtuple('PPCParameters', [
            'hue_value',
            'hue_width',
            'saturation_minimum',
            'intensity_upper_limit',
            'intensity_weak_threshold',
            'intensity_strong_threshold',
            'intensity_lower_limit',
        ]),
):
    """PPCParameters(hue_value, hue_width, saturation_minimum,
    intensity_upper_limit, intensity_weak_threshold,
    intensity_strong_threshold, intensity_lower_limit)

    Attributes:
    - hue_value: center of the hue range in HSI space for the positive
      color, in the range [0, 1]
    - hue_width: width of the hue range in HSI space
    - saturation_minimum: minimum saturation of positive pixels in HSI
      space, in the range [0, 1]
    - intensity_upper_limit: intensity threshold in HSI space above which
      a pixel is considered negative, in the range [0, 1]
    - intensity_weak_threshold: intensity threshold in HSI space that
      separates weak-positive pixels (above) from plain positive pixels
      (below)
    - intensity_strong_threshold: intensity threshold in HSI space that
      separates plain positive pixels (above) from strong positive pixels
      (below)
    - intensity_lower_limit: intensity threshold in HSI space below which
      a pixel is considered negative

    """


PPCOutputTotals = namedtuple('PPCOutputTotals', [
    'NumberWeakPositive',
    'NumberPositive',
    'NumberStrongPositive',
    'IntensitySumWeakPositive',
    'IntensitySumPositive',
    'IntensitySumStrongPositive',
])

PPCOutput = namedtuple('PPCOutput', PPCOutputTotals._fields + (
    'IntensityAverage',
    'RatioStrongToTotal',
    'IntensityAverageWeakAndPositive',
))


def positive_pixel_count(slide_path, ppc_params, region=None,
                         tile_grouping=256, make_label_image=False):
    """Compute a count of positive pixels in the slide at slide_path.
    This routine can also create a label image.

    Parameters
    ---------
    slide_path : string (path)
        Path to the slide to analyze.
    ppc_params : PPCParameters
        An instance of PPCParameters, which see for further documentation
    region : dict, optional
        A valid region dict (per a large_image
        TileSource.tileIterator's region argument)
    tile_grouping : int
        The number of tiles to process as part of a single task
    make_label_image : bool, default=False
        Whether to make a label image.  See also "Notes"

    Returns
    -------
    stats : PPCOutput
        Various statistics on the input image.  See PPCOutput.
    label_image : array-like, only if make_label_image is set

    Notes
    -----
    The return value is either a single or a pair -- it is in either
    case a tuple.  Dask is used as configured to compute the
    statistics, but only if make_label_image is reset.  If
    make_label_image is set, everything is computed in a
    single-threaded manner.

    """
    ts = large_image.getTileSource(slide_path)
    kwargs = dict(format=large_image.tilesource.TILE_FORMAT_NUMPY)
    if region is not None:
        kwargs['region'] = region
    if make_label_image:
        tile = ts.getRegion(**kwargs)[0]
        return positive_pixel_count_simple(tile, ppc_params)
    else:
        results = []
        total_tiles = ts.getSingleTile(**kwargs)['iterator_range']['position']
        for position in range(0, total_tiles, tile_grouping):
            results.append(delayed(positive_pixel_count_tiles)(
                slide_path, ppc_params, kwargs, position,
                min(tile_grouping, total_tiles - position)))
        results = delayed(_combine)(results).compute()
    return _totals_to_stats(results),


def _combine(results):
    return PPCOutputTotals._make(sum(r[i] for r in results)
                                 for i in range(len(PPCOutputTotals._fields)))


def positive_pixel_count_tiles(slide_path, ppc_params, kwargs, position, count):
    ts = large_image.getTileSource(slide_path)
    lpotf = len(PPCOutputTotals._fields)
    total = [0] * lpotf
    for pos in range(position, position + count):
        tile = ts.getSingleTile(tile_position=pos, **kwargs)['tile']
        subtotal = _positive_pixel_count_simple(tile, ppc_params)[0]
        for k in range(lpotf):
            total[k] += subtotal[k]
    return PPCOutputTotals._make(total)


def positive_pixel_count_simple(image, parameters):
    """Count positive pixels, computing a label mask and summary
    statistics.

    Parameters
    ----------
    image : array-like
        NxMx3 array of RGB data
    parameters : PPCParameters
        An instance of PPCParameters, which see for further documentation

    Returns
    -------
    stats : PPCOutput
        Various statistics on the input image.  See PPCOutput.
    label_image : array-like
        NxM array of pixel types.  See Labels for the different values.

    """
    total, masks = _positive_pixel_count_simple(image, parameters)
    mask_all_positive, mask_weak, mask_pos, mask_strong = masks
    label_image = np.full(image.shape[:-1], Labels.NEGATIVE, dtype=np.uint8)
    label_image[mask_all_positive] = (
        mask_weak * Labels.WEAK +
        mask_pos * Labels.PLAIN +
        mask_strong * Labels.STRONG
    )
    return _totals_to_stats(total), label_image


def _positive_pixel_count_simple(image, parameters):
    """A version of positive_pixel_count_simple that doesn't compute the
    label image and only computes the sums.

    """
    p = parameters
    image_hsi = rgb_to_hsi(image / 255)
    mask_all_positive = (
        (np.abs((image_hsi[..., 0] - p.hue_value + 0.5 % 1) - 0.5) <=
         p.hue_width / 2) &
        (image_hsi[..., 1] >= p.saturation_minimum) &
        (image_hsi[..., 2] < p.intensity_upper_limit) &
        (image_hsi[..., 2] >= p.intensity_lower_limit)
    )
    all_positive_i = image_hsi[mask_all_positive, 2]
    mask_weak = all_positive_i >= p.intensity_weak_threshold
    nw, iw = np.count_nonzero(mask_weak), np.sum(all_positive_i[mask_weak])
    mask_strong = all_positive_i < p.intensity_strong_threshold
    ns, is_ = np.count_nonzero(mask_strong), np.sum(all_positive_i[mask_strong])
    mask_pos = ~(mask_weak | mask_strong)
    np_, ip = np.count_nonzero(mask_pos), np.sum(all_positive_i[mask_pos])
    total = PPCOutputTotals(
        NumberWeakPositive=nw,
        NumberPositive=np_,
        NumberStrongPositive=ns,
        IntensitySumWeakPositive=iw,
        IntensitySumPositive=ip,
        IntensitySumStrongPositive=is_,
    )
    return total, (mask_all_positive, mask_weak, mask_pos, mask_strong)


def _totals_to_stats(total):
    """Do the extra computations to convert a PPCOutputTotals to a PPCOutput"""
    t = total
    all_positive = t.NumberWeakPositive + t.NumberPositive + t.NumberStrongPositive
    return PPCOutput(
        IntensityAverage=((t.IntensitySumWeakPositive
                           + t.IntensitySumPositive
                           + t.IntensitySumStrongPositive)
                          / all_positive),
        RatioStrongToTotal=t.NumberStrongPositive / all_positive,
        IntensityAverageWeakAndPositive=(
            (t.IntensitySumWeakPositive + t.IntensitySumPositive)
            / (t.NumberWeakPositive + t.NumberPositive)
        ),
        **t._asdict()
    )


__all__ = (
    'PPCParameters',
    'PPCOutput',
    'positive_pixel_count',
    'positive_pixel_count_simple',
)
