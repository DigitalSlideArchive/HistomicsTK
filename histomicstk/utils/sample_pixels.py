import large_image
import numpy as np
import PIL.Image

from .simple_mask import simple_mask


def sample_pixels(slide_path, sample_fraction=None, magnification=None,
                  tissue_seg_mag=1.25, min_coverage=0.1, background=False,
                  sample_approximate_total=None, tile_grouping=256, invert_image=False,
                  style=None, frame=None, default_img_inversion=False):
    """Generates a sampling of pixels from a whole-slide image.

    Useful for generating statistics or Reinhard color-normalization or
    adaptive deconvolution. Uses mixture modeling approach to focus
    sampling in tissue regions.

    Parameters
    ----------
    slide_path : str
        path and filename of slide.
    sample_fraction : double
        Fraction of pixels to sample. Must be in the range [0, 1].
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
        use instead of sample_fraction to specify roughly how many pixels to
        sample. The fewer tiles are excluded, the more accurate this will be.
    tile_grouping: int, optional
        Number of tiles to process as part of a single task.

    Returns
    -------
    pixels : array_like
        A Nx3 matrix of RGB pixel values sampled from the whole-slide.

    Notes
    -----
    If Dask is configured, it is used to distribute the computation.

    See Also
    --------
    histomicstk.preprocessing.color_normalization.reinhard

    """
    import dask
    import dask.distributed

    if (sample_fraction is None) == (sample_approximate_total is None):
        msg = 'Exactly one of sample_fraction and sample_approximate_total must have a value.'
        raise ValueError(msg)

    ts = large_image.getTileSource(slide_path, style=style)

    if magnification is None:
        magnification = ts.getMetadata()['magnification']

    # get entire whole-slide image at low resolution
    scale_lres = {'magnification': tissue_seg_mag}
    im_lres, _ = ts.getRegion(
        format=large_image.tilesource.TILE_FORMAT_NUMPY,
        scale=scale_lres,
        frame=frame,
    )

    # check number of channels
    if len(im_lres.shape) <= 2 or im_lres.shape[2] == 1:
        im_lres = np.dstack((im_lres, im_lres, im_lres))
        if default_img_inversion:
            invert_image = True
    else:
        im_lres = im_lres[:, :, :3]

    # perform image inversion
    if invert_image:
        im_lres = (np.iinfo(im_lres.dtype).max
                   if im_lres.dtype.kind == 'u'
                   else np.max(im_lres)) - im_lres

    # compute foreground mask of whole-slide image at low-res.
    # it will actually be a background mask if background is set.
    im_fgnd_mask_lres = bool(background) ^ simple_mask(im_lres)

    if sample_approximate_total is not None:
        scale_ratio = float(magnification) / tissue_seg_mag
        total_fgnd_pixels = np.count_nonzero(im_fgnd_mask_lres) * scale_ratio ** 2
        sample_fraction = sample_approximate_total / total_fgnd_pixels

    # broadcasting fgnd mask to all dask workers
    try:
        c = dask.distributed.get_client()

        [im_fgnd_mask_lres] = c.scatter([im_fgnd_mask_lres],
                                        broadcast=True)
    except ValueError:
        pass

    # generate sample pixels
    sample_pixels = []

    iter_args = dict(scale=dict(magnification=magnification),
                     format=large_image.tilesource.TILE_FORMAT_NUMPY)

    total_tiles = ts.getSingleTile(**iter_args)['iterator_range']['position']

    for position in range(0, total_tiles, tile_grouping):

        sample_pixels.append(dask.delayed(_sample_pixels_tile)(
            slide_path, iter_args,
            (position, min(tile_grouping, total_tiles - position)),
            sample_fraction, tissue_seg_mag, min_coverage,
            im_fgnd_mask_lres, invert_image=invert_image, style=style,
            default_img_inversion=default_img_inversion))

    # concatenate pixel values in list
    if sample_pixels:
        sample_pixels = (dask.delayed(np.concatenate)(sample_pixels, 0)
                         .compute())
    else:
        print('Sampling could not identify any foreground regions.')

    return sample_pixels


def _sample_pixels_tile(slide_path, iter_args, positions, sample_fraction,
                        tissue_seg_mag, min_coverage, im_fgnd_mask_lres,
                        invert_image=False, style=None, default_img_inversion=False):
    start_position, position_count = positions
    sample_pixels = [np.empty((0, 3))]
    ts = large_image.getTileSource(slide_path, style=style)
    for position in range(start_position, start_position + position_count):
        tile = ts.getSingleTile(tile_position=position, **iter_args)
        # get current region in base_pixels
        rgn_hres = {'left': tile['gx'], 'top': tile['gy'],
                    'right': tile['gx'] + tile['gwidth'],
                    'bottom': tile['gy'] + tile['gheight'],
                    'units': 'base_pixels'}

        # get foreground mask for current tile at low resolution
        rgn_lres = ts.convertRegionScale(rgn_hres,
                                         targetScale={'magnification':
                                                      tissue_seg_mag},
                                         targetUnits='mag_pixels')

        top = int(rgn_lres['top'])
        bottom = int(rgn_lres['bottom'])
        left = int(rgn_lres['left'])
        right = int(rgn_lres['right'])

        tile_fgnd_mask_lres = im_fgnd_mask_lres[top:bottom, left:right]

        # skip tile if there is not enough foreground in the slide
        cur_fgnd_frac = tile_fgnd_mask_lres.mean()

        if np.isnan(cur_fgnd_frac) or cur_fgnd_frac <= min_coverage:
            continue

        # check number of channels
        if len(tile['tile'].shape) <= 2 or tile['tile'].shape[2] == 1:
            im_tile = np.dstack((tile['tile'], tile['tile'], tile['tile']))
            if default_img_inversion:
                invert_image = True

        else:
            im_tile = tile['tile'][:, :, :3]

        # perform image inversion
        if invert_image:
            im_tile = (np.iinfo(im_tile.dtype).max
                       if im_tile.dtype.kind == 'u'
                       else np.max(im_tile)) - im_tile
        # get tile foreground mask at resolution of current tile
        tile_fgnd_mask = np.array(PIL.Image.fromarray(tile_fgnd_mask_lres).resize(
            im_tile.shape[:2],
            resample=PIL.Image.NEAREST,
        ))

        # generate linear indices of sample pixels in fgnd mask
        nz_ind = np.nonzero(tile_fgnd_mask.flatten())[0]

        # Handle fractions in the desired sample size by rounding up
        # or down, weighted by the fractional amount.
        float_samples = sample_fraction * nz_ind.size
        num_samples = int(np.floor(float_samples))
        num_samples += np.random.binomial(1, float_samples - num_samples)

        sample_ind = np.random.choice(nz_ind, num_samples)

        # convert rgb tile image to Nx3 array
        tile_pix_rgb = np.reshape(im_tile, (-1, 3))

        # add rgb triplet of sample pixels
        sample_pixels.append(tile_pix_rgb[sample_ind, :])

    return np.concatenate(sample_pixels, 0)
