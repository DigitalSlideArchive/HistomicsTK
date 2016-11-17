import numpy as np
import scipy as sp


def clog(im_input, mask, sigma_min=30 * 1.414, sigma_max=50 * 1.414):
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
    mask : array_like
        A binary image where nuclei pixels have value 1/True, and non-nuclear
        pixels have value 0/False.
    sigma_min : float
        A scalar defining the minimum scaled nuclear radius. Radius is scaled
        by sqrt(2). Default value = 30 * 2 ** 0.5.
    sigma_max : float
        A scalar defining the maximum scaled nuclear radius. Radius is scaled
        by sqrt(2). Default value = 50 * 2 ** 0.5.

    Returns
    -------
    im_log_max : array_like
        An intensity image containing the maximal LoG filter response accross
        all scales for each pixel

    References
    ----------
    .. [1] Y. Al-Kofahi et al "Improved Automatic Detection and Segmentation
           of Cell Nuclei in Histopathology Images" in IEEE Transactions on
           Biomedical Engineering,vol.57,no.4,pp.847-52, 2010.
    """

    # convert intensity image type to float if needed
    if im_input.dtype == np.uint8:
        im_input = im_input.astype(np.float)

    # generate distance map
    Distance = sp.ndimage.morphology.distance_transform_edt(mask)

    # initialize constraint
    Constraint = np.maximum(sigma_min, np.minimum(sigma_max, 2 * Distance))

    # initialize log filter response array
    im_log_max = np.finfo(Distance.dtype).min * np.ones(mask.shape)

    # LoG filter over scales
    Start = np.floor(sigma_min)
    Stop = np.ceil(sigma_max)
    Sigmas = np.linspace(Start, Stop, Stop-Start+1)
    for Sigma in Sigmas:

        # generate normalized filter response
        Response = Sigma ** 2 * \
            sp.ndimage.filters.gaussian_laplace(im_input, Sigma, mode='mirror')

        # constrain response
        Map = Sigma < Constraint
        Response[~Map] = np.finfo(Distance.dtype).min

        # replace with maxima
        im_log_max = np.maximum(im_log_max, Response)

    # translate filtered image

    # replace min floats
    im_log_max[im_log_max == np.finfo(Distance.dtype).min] = 0

    return im_log_max
