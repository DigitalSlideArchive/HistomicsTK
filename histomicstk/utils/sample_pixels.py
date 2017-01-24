import large_image
import numpy as np
import scipy

from .simple_mask import simple_mask


def sample_pixels(slide_path, magnification, sample_percent,
                  tissue_seg_mag=1.25, min_coverage=0.1):
    """Generates a sampling of pixels from a whole-slide image.

    Useful for generating statistics or Reinhard color-normalization or
    adaptive deconvolution. Uses mixture modeling approach to focus
    sampling in tissue regions.

    Parameters
    ----------
    slide_path : str
        path and filename of slide.
    magnification : double
        Desired magnification for sampling (defaults to native scan
        magnification).
    sample_percent : double
        Percentage of pixels to sample. Must be in the range [0, 1].
    tissue_seg_mag: double, optional
        low resolution magnification at which foreground will be segmented.
        Default value = 1.25.
    min_coverage: double, optional
        minimum sample_percent of tile covered by tissue to be included in sampling.
        Ranges between [0,1). Default value = 0.1.

    Returns
    -------
    Pixels : array_like
        A Nx3 matrix of RGB pixel values sampled from the whole-slide.

    See Also
    --------
    histomicstk.preprocessing.color_normalization.reinhard
    """

    ts = large_image.getTileSource(slide_path)

    # get enitre whole-silde image at low resolution
    scale_lres = {'magnification': tissue_seg_mag}
    im_lres, _ = ts.getRegion(
        format=large_image.tilesource.TILE_FORMAT_NUMPY,
        scale=scale_lres
    )
    im_lres = im_lres[:, :, :3]

    # compute foreground mask of whole-slide image at low-res
    im_fgnd_mask_lres = simple_mask(im_lres)

    # generate sample pixels
    sample_pixels = []

    scale_hres = {'magnfication': magnification}

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
        sample_ind = np.random.choice(nz_ind,
                                      np.ceil(sample_percent * nz_ind.size))

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
