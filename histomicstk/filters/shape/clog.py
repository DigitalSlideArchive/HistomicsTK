import numpy as np


def clog(im_input, im_mask, sigma_min, sigma_max):
    """Constrainted Laplacian of Gaussian filter.

    Takes as input a grayscale nuclear image and binary mask of cell nuclei,
    and uses the distance transform of the nuclear mask to constrain the LoG
    filter response of the image for nuclear seeding. Returns a LoG filter
    image of type float. Local maxima are used for seeding cells.

    Parameters
    ----------
    im_input : array_like
        A hematoxylin intensity image obtained from ColorDeconvolution. Objects
        are assumed to be dark with a light background.
    im_mask : array_like
        A binary image where nuclei pixels have value 1/True, and non-nuclear
        pixels have value 0/False.
    sigma_min : double
        Minumum sigma value for the scale space. For blob detection, set this
        equal to minimum-blob-radius / sqrt(2).
    sigma_max : double
        Maximum sigma value for the scale space. For blob detection, set this
        equal to maximum-blob-radius / sqrt(2).

    Returns
    -------
    im_log_max : array_like
        An intensity image containing the maximal LoG filter response accross
        all scales for each pixel
    im_sigma_max : array_like
        An intensity image containing the sigma value corresponding to the
        maximal LoG response at each pixel. The nuclei/blob radius value for
        a given sigma can be estimated to be equal to sigma * sqrt(2).

    References
    ----------
    .. [#] Y. Al-Kofahi et al "Improved Automatic Detection and Segmentation
           of Cell Nuclei in Histopathology Images" in IEEE Transactions on
           Biomedical Engineering,vol.57,no.4,pp.847-52, 2010.

    """
    from scipy.ndimage.filters import gaussian_laplace
    from scipy.ndimage.morphology import distance_transform_edt

    # convert intensity image type to float
    im_input = im_input.astype(float)

    # generate distance map
    im_dmap = distance_transform_edt(im_mask)

    # compute max sigma at each pixel as 2 times the distance to background
    im_sigma_ubound = 2.0 * im_dmap

    # clip max sigma values to specified range
    im_sigma_ubound = np.clip(im_sigma_ubound, sigma_min, sigma_max)

    # initialize log filter response array
    MIN_FLOAT = np.finfo(im_input.dtype).min

    im_log_max = np.zeros_like(im_input)
    im_log_max[:, :] = MIN_FLOAT

    im_sigma_max = np.zeros_like(im_input)

    # Compute maximal LoG filter response across the scale space
    sigma_start = int(np.floor(sigma_min))
    sigma_end = int(np.ceil(sigma_max))

    sigma_list = np.linspace(sigma_start, sigma_end,
                             int(sigma_end - sigma_start + 1))

    for sigma in sigma_list:

        # generate normalized filter response
        im_log_cur = sigma**2 * gaussian_laplace(im_input, sigma, mode='mirror')

        # constrain LoG response
        im_log_cur[im_sigma_ubound < sigma] = MIN_FLOAT

        # update maxima
        max_update_pixels = np.where(im_log_cur > im_log_max)

        if len(max_update_pixels[0]) > 0:

            im_log_max[max_update_pixels] = im_log_cur[max_update_pixels]
            im_sigma_max[max_update_pixels] = sigma

    # replace min floats
    im_log_max[im_log_max == MIN_FLOAT] = 0

    return im_log_max, im_sigma_max
