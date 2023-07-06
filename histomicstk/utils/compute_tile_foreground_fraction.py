import large_image
import numpy as np


def compute_tile_foreground_fraction(slide_path, im_fgnd_mask_lres,
                                     fgnd_seg_scale, it_kwargs,
                                     tile_position=None, style=None):
    """
    Computes the fraction of foreground of a single tile or
    all tiles in a whole slide image given the binary foreground
    mask computed from a low resolution version of the slide.

    Parameters
    ----------
    slide_path : str
        path to an image or slide
    im_fgnd_mask_lres : array_like
        A binary foreground mask computed at a low-resolution
    fgnd_seg_scale : double
        The scale/magnification at which the foreground mask `im_fgnd_mask_lres`
        was computed
    it_kwargs : dict
        A dictionary of any key:value parameters (e.g. defining the scale,
         tile_size, region etc) in addition to tile_position that need to be
         passed to `large_image.TileSource.getSingleTile` to get the tile.
    tile_position : int or None
        A linear 0-based index of a tile for which the foreground needs to be
        computed. If set to None, the foreground fraction of all tiles will be
        computed.

    Returns
    -------
    tile_fgnd_frac : double or array_like
        A value between 0 and 1 indicating the fraction of foreground pixels
        present in the tile indicated by `tile_position`. If `tile_position`
        is set to None, then a 1D array containing the foreground fraction of
        all tiles will be returned.

    """
    import dask
    import dask.distributed

    if tile_position is None:

        # get slide tile source
        ts = large_image.getTileSource(slide_path, style=style)

        num_tiles = ts.getSingleTile(**it_kwargs)['iterator_range']['position']

        # broadcasting fgnd mask to all dask workers
        try:
            c = dask.distributed.get_client()

            [im_fgnd_mask_lres] = c.scatter([im_fgnd_mask_lres],
                                            broadcast=True)
        except ValueError:
            pass

        # compute tile foreground fraction in parallel
        tile_fgnd_frac = []

        for tile_position in range(num_tiles):

            tile_fgnd_frac.append(
                dask.delayed(_compute_tile_foreground_fraction_single)(
                    slide_path, im_fgnd_mask_lres, fgnd_seg_scale,
                    it_kwargs, tile_position
                ))

        tile_fgnd_frac = dask.delayed(tile_fgnd_frac).compute()

        tile_fgnd_frac = np.array(tile_fgnd_frac)

    elif np.isscalar(tile_position):

        tile_fgnd_frac = _compute_tile_foreground_fraction_single(
            slide_path, im_fgnd_mask_lres, fgnd_seg_scale,
            it_kwargs, tile_position, style=style
        )

    else:

        raise ValueError(
            'Invalid value for tile_position. Must be None or int')

    return tile_fgnd_frac


def _compute_tile_foreground_fraction_single(slide_path, im_fgnd_mask_lres,
                                             fgnd_seg_scale, it_kwargs,
                                             tile_position, style=None):

    # get slide tile source
    ts = large_image.getTileSource(slide_path, style=style)

    # get requested tile
    tile = ts.getSingleTile(tile_position=tile_position,
                            format=large_image.tilesource.TILE_FORMAT_NUMPY,
                            **it_kwargs)

    # get current region in base_pixels
    rgn_hres = {'left': tile['gx'], 'top': tile['gy'],
                'right': tile['gx'] + tile['gwidth'],
                'bottom': tile['gy'] + tile['gheight'],
                'units': 'base_pixels'}

    # get foreground mask for current tile at low resolution
    rgn_lres = ts.convertRegionScale(rgn_hres,
                                     targetScale=fgnd_seg_scale,
                                     targetUnits='mag_pixels')

    top = int(rgn_lres['top'])
    bottom = int(rgn_lres['bottom'])
    left = int(rgn_lres['left'])
    right = int(rgn_lres['right'])

    im_tile_fgnd_mask_lres = im_fgnd_mask_lres[top:bottom, left:right]

    # compute foreground fraction
    tile_fgnd_frac = im_tile_fgnd_mask_lres.mean()

    if np.isnan(tile_fgnd_frac):
        tile_fgnd_frac = 0

    return tile_fgnd_frac
