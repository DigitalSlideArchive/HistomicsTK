import large_image
import numpy as np
import scipy

from .simple_mask import simple_mask


def sample_pixels(slide_path, magnification, percent, tile_size,
                  mapping_mag=1.25, min_coverage=0.1):
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
    percent : double
        Percentage of pixels to sample. Must be in the range [0, 1].
    tile_size : int
        tile_size size used in sampling high-resolution image.
    mapping_mag: double, optional
        low resolution magnification. Default value = 1.25.
    min_coverage: double, optional
        minimum percent of tile covered by tissue to be included in sampling.
        Ranges between [0,1). Default value = 0.1.

    Returns
    -------
    Pixels : array_like
        A 3xN matrix of RGB pixel values sampled from the whole-slide.

    See Also
    --------
    histomicstk.preprocessing.color_normalization.reinhard,
    histomicstk.preprocessing.color_deconvolution.SparseColorDeconvolution
    """

    ts = large_image.getTileSource(slide_path)

    # get enitre whole-silde image at low resolution
    scale_lowres = {'magnification': mapping_mag}
    im_lowres, _ = ts.getRegion(
        format=large_image.tilesource.TILE_FORMAT_NUMPY,
        scale=scale_lowres
    )

    # compute foreground mask of whole-slide image at low-res
    fgnd_mask_lowres = simple_mask(im_lowres)

    # generate sample pixels
    sample_pixels = []

    scale_highres = {'magnfication': magnification}

    for tile in ts.tileIterator(
            scale=scale_highres,
            format=large_image.tilesource.TILE_FORMAT_NUMPY):

        # get current tile image
        im_tile = tile['tile'][:, :, :3]

        # get current region in base_pixels
        rgn_hres = {'left': tile['gx'], 'top': tile['gy'],
                    'width': tile['gwidth'], 'height': tile['gheight'],
                    'units': 'base_pixels'}

        # get foreground mask for current tile at low resolution
        rgn_lres = ts.convertRegionScale(rgn_hres, target_scale=scale_lowres)

        tile_fgnd_mask_lowres = \
            fgnd_mask_lowres[rgn_lres['left']: rgn_lres['right'],
                             rgn_lres['top']: rgn_lres['bottom']]

        # skip tile if there is not enough foreground in the slide
        if tile_fgnd_mask_lowres.mean() < min_coverage:
            continue

        # get tile foreground mask at resolution of current tile
        tile_fgnd_mask = scipy.misc.imresize(
            tile_fgnd_mask_lowres,
            im_tile.shape,
            interp='nearest'
        )

        # generate linear indices of sample pixels in fgnd mask
        nz_ind = np.nonzero(tile_fgnd_mask.flatten())[0]
        sample_ind = np.random.choice(nz_ind, np.ceil(percent * nz_ind.size))

        # convert rgb tile image to 3xN array
        tile_pix_rgb = np.reshape(im_tile,
                                  (im_tile.shape[0] * im_tile.shape[1], 3))

        # add rgb triplet of sample pixels
        sample_pixels.append(tile_pix_rgb[sample_ind, :])

    # concatenate pixel values in list
    try:
        sample_pixels = np.concatenate(sample_pixels, 1)
    except ValueError:
        print "Sampling could not identify any foreground regions."

    return sample_pixels
