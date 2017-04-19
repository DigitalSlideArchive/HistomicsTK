import large_image
import numpy as np
import scipy

from .simple_mask import simple_mask


def sample_pixels(slide_path, sample_percent=None, magnification=None,
                  tissue_seg_mag=1.25, min_coverage=0.1, background=False,
                  sample_approximate_total=None):
    """Generates a sampling of pixels from a whole-slide image.

    Useful for generating statistics or Reinhard color-normalization or
    adaptive deconvolution. Uses mixture modeling approach to focus
    sampling in tissue regions.

    Parameters
    ----------
    slide_path : str
        path and filename of slide.
    sample_percent : double
        Percentage of pixels to sample. Must be in the range [0, 1].
    magnification : double
        Desired magnification for sampling.
        Default value : None (for native scan magnification).
    tissue_seg_mag: double, optional
        low resolution magnification at which foreground will be segmented.
        Default value = 1.25.
    min_coverage: double, optional
        minimum fraction of tile covered by tissue for it to be included
        in sampling. Ranges between [0,1). Default value = 0.1.
    background: bool, optional
        sample the background instead of the foreground if True. min_coverage
        then refers to the amount of background. Default value = False
    sample_approximate_total: int, optional
        use instead of sample_percent to specify roughly how many pixels to
        sample. The fewer tiles are excluded, the more accurate this will be.

    Returns
    -------
    Pixels : array_like
        A Nx3 matrix of RGB pixel values sampled from the whole-slide.

    See Also
    --------
    histomicstk.preprocessing.color_normalization.reinhard
    """

    if (sample_percent is None) == (sample_approximate_total is None):
        raise ValueError('Exactly one of sample_percent and ' +
                         'sample_approximate_total must have a value.')

    ts = large_image.getTileSource(slide_path)

    if magnification is None:
        magnification = ts.getMetadata()['magnification']

    # get entire whole-slide image at low resolution
    scale_lres = {'magnification': tissue_seg_mag}
    im_lres, _ = ts.getRegion(
        format=large_image.tilesource.TILE_FORMAT_NUMPY,
        scale=scale_lres
    )
    im_lres = im_lres[:, :, :3]

    # compute foreground mask of whole-slide image at low-res.
    # it will actually be a background mask if background is set.
    im_fgnd_mask_lres = bool(background) ^ simple_mask(im_lres)

    if sample_approximate_total is not None:
        scale_ratio = float(magnification) / tissue_seg_mag
        total_fgnd_pixels = np.count_nonzero(im_fgnd_mask_lres) * scale_ratio ** 2
        sample_percent = sample_approximate_total / total_fgnd_pixels

    # generate sample pixels
    sample_pixels = []

    scale_hres = {'magnification': magnification}

    # Accumulator for probabilistic rounding
    frac_accum = 0.

    for tile in ts.tileIterator(
            scale=scale_hres,
            format=large_image.tilesource.TILE_FORMAT_NUMPY):

        # get current region in base_pixels
        rgn_hres = {'left': tile['gx'], 'top': tile['gy'],
                    'right': tile['gx'] + tile['gwidth'],
                    'bottom': tile['gy'] + tile['gheight'],
                    'units': 'base_pixels'}

        # get foreground mask for current tile at low resolution
        rgn_lres = ts.convertRegionScale(rgn_hres,
                                         targetScale=scale_lres,
                                         targetUnits='mag_pixels')

        top = np.int(rgn_lres['top'])
        bottom = np.int(rgn_lres['bottom'])
        left = np.int(rgn_lres['left'])
        right = np.int(rgn_lres['right'])

        tile_fgnd_mask_lres = im_fgnd_mask_lres[top:bottom, left:right]

        # skip tile if there is not enough foreground in the slide
        cur_fgnd_frac = tile_fgnd_mask_lres.mean()

        if np.isnan(cur_fgnd_frac) or cur_fgnd_frac <= min_coverage:
            continue

        # get current tile image
        im_tile = tile['tile'][:, :, :3]

        # get tile foreground mask at resolution of current tile
        tile_fgnd_mask = scipy.misc.imresize(
            tile_fgnd_mask_lres,
            im_tile.shape,
            interp='nearest'
        )

        # generate linear indices of sample pixels in fgnd mask
        nz_ind = np.nonzero(tile_fgnd_mask.flatten())[0]

        # Handle fractions in the desired sample size by rounding up
        # or down, weighted by the fractional amount.  To reduce
        # variance, the fraction is adjusted by a running counter that
        # factors in previous random decisions.
        float_samples = sample_percent * nz_ind.size
        num_samples = int(np.floor(float_samples))
        frac_accum += float_samples - num_samples
        r = np.random.binomial(1, np.clip(frac_accum, 0, 1))
        num_samples += r
        frac_accum -= r

        sample_ind = np.random.choice(nz_ind, num_samples)

        # convert rgb tile image to Nx3 array
        tile_pix_rgb = np.reshape(im_tile,
                                  (im_tile.shape[0] * im_tile.shape[1], 3))

        # add rgb triplet of sample pixels
        sample_pixels.append(tile_pix_rgb[sample_ind, :])

    # concatenate pixel values in list
    try:
        sample_pixels = np.concatenate(sample_pixels, 0)
    except ValueError:
        print "Sampling could not identify any foreground regions."

    return sample_pixels
