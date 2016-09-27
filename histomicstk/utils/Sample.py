import large_image
import numpy as np
import scipy

from .SimpleMask import SimpleMask


def Sample(slide_path, magnification, percent, tile_size,
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
        A 3xN matrix of RGB pixel values sampled from the slide at `slide_path`.

    See Also
    --------
    histomicstk.preprocessing.color_normalization.reinhard,
    histomicstk.preprocessing.color_deconvolution.SparseColorDeconvolution
    """

    ts = large_image.getTileSource(slide_path)

    # get enitre whole-silde image at low resolution
    im_lowres, _ = ts.getRegion(
        scale={'magnification': mapping_mag},
        format=large_image.tilesource.TILE_FORMAT_NUMPY
    )

    # compute foreground mask of whole-slide image at low-res
    fgnd_mask_lowres = SimpleMask(im_lowres)

    # generate sample pixels
    sample_pixels = []

    for tile in ts.tileIterator(
            format=large_image.tilesource.TILE_FORMAT_NUMPY,
            scale={'magnification': magnification}):

        # get current tile image
        im_tile = tile['tile'][:, :, :3]

        # get fgnd mask for current tile
        mask_scale = mapping_mag / magnification

        left = np.round(tile['gx'] * mask_scale)
        top = np.round(tile['gy'] * mask_scale)

        right = np.round(left + tile['gwidth'] * mask_scale)
        bottom = np.round(top + tile['gheight'] * mask_scale)

        tile_fgnd_mask = scipy.misc.imresize(
            fgnd_mask_lowres[left: right, top: bottom],
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
        sample_pixels = np.concatenate(Pixels, 1)
    except ValueError:
        print "Sampling could not identify any foreground regions."

    return sample_pixels
