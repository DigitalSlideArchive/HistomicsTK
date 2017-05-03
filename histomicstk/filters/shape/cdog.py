import math

import numpy as np

from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize


def cdog(im_input, im_mask, sigma_min, sigma_max, num_octave_levels=3):
    """SCale-adaptive Multiscale Difference-of-Gaussian (DoG) filter for
    nuclei/blob detection.

    Computes the maximal DoG response over a series of scales where in the
    applicable scales at each pixel are constrained to be below an upper-bound
    equal to 2 times the distance to the nearest non-nuclear/background pixel.

    This function uses an approach similar to SIFT interest detection
    where in the scale space between the specified min and max sigma values is
    divided into octaves (scale/sigma is doubled after each octave) and each
    octave is divided into sub-levels. The gaussian images are downsampled by 2
    at the end of each octave to keep the size of convolutional filters small.

    Parameters
    ----------
    im_input : array_like
        A hematoxylin intensity image obtained from ColorDeconvolution. Objects
        are assumed to be dark with a light background.
    mask : array_like
        A binary image where nuclei pixels have non-zero values
    sigma_min : double
        Minumum sigma value for the scale space. For blob detection, set this
        equal to minimum-blob-radius / sqrt(2).
    sigma_max : double
        Maximum sigma value for the scale space. For blob detection, set this
        equal to maximum-blob-radius / sqrt(2).
    num_octave_levels : int
        Number of levels per octave in the scale space.

    Returns
    -------
    im_dog_max : array_like
        An intensity image containing the maximal DoG response accross
        all scales for each pixel
    im_sigma_max : array_like
        An intensity image containing the sigma value corresponding to the
        maximal LoG response at each pixel. The nuclei/blob radius for
        a given sigma value can be calculated as sigma * sqrt(2).


    References
    ----------
    .. [*] D.G. Lowe "Distinctive image features from scale-invariant
           keypoints." International journal of computer vision, vol. 60,
           no. 2, 91-110, 2004.

    """

    im_input = im_input.astype(np.float)

    # generate distance map
    im_dmap = distance_transform_edt(im_mask)

    # compute max sigma at each pixel as 2 times the distance to background
    im_sigma_ubound = 2.0 * im_dmap

    # clip max sigma values to specified range
    im_sigma_ubound = np.clip(im_sigma_ubound, sigma_min, sigma_max)

    # compute number of levels in the scale space
    sigma_ratio = 2 ** (1.0 / num_octave_levels)

    k = int(math.log(float(sigma_max) / sigma_min, sigma_ratio)) + 1

    # Compute maximal DoG filter response accross the scale space
    sigma_cur = sigma_min
    im_gauss_cur = gaussian_filter(im_input, sigma_cur)
    im_sigma_ubound_cur = im_sigma_ubound.copy()

    MIN_FLOAT = np.finfo(im_input.dtype).min

    im_dog_max = np.zeros_like(im_input)
    im_dog_max[:, :] = MIN_FLOAT
    im_dog_octave_max = im_dog_max.copy()

    im_sigma_max = np.zeros_like(im_input)
    im_sigma_octave_max = np.zeros_like(im_input)

    n_level = 0
    n_octave = 0

    for i in range(k + 1):

        # calculate sigma at next level
        sigma_next = sigma_cur * sigma_ratio

        # Do cascaded convolution to keep convolutional kernel small
        # G(a) * G(b) = G(sqrt(a^2 + b^2))
        sigma_conv = np.sqrt(sigma_next ** 2 - sigma_cur ** 2)
        sigma_conv /= 2.0 ** n_octave

        im_gauss_next = gaussian_filter(im_gauss_cur, sigma_conv)

        # compute DoG
        im_dog_cur = im_gauss_next - im_gauss_cur

        # constrain response
        im_dog_cur[im_sigma_ubound_cur < sigma_cur] = MIN_FLOAT

        # update maxima
        max_update_pixels = np.where(im_dog_cur > im_dog_octave_max)

        if len(max_update_pixels[0]) > 0:

            im_dog_octave_max[max_update_pixels] = im_dog_cur[max_update_pixels]
            im_sigma_octave_max[max_update_pixels] = sigma_cur

            # print np.unique(im_sigma_octave_max)

        # update cur sigma
        sigma_cur = sigma_next
        im_gauss_cur = im_gauss_next

        # udpate level
        n_level += 1

        # Do additional processing at the end of each octave
        if i == k or n_level == num_octave_levels:

            # update maxima
            if num_octave_levels > 0:

                im_dog_octave_max_rszd = resize(
                    im_dog_octave_max, im_dog_max.shape, order=0)

            else:

                im_dog_octave_max_rszd = im_dog_octave_max

            max_pixels = np.where(
                im_dog_octave_max_rszd > im_dog_max)

            if len(max_pixels[0]) > 0:

                im_dog_max[max_pixels] = \
                    im_dog_octave_max_rszd[max_pixels]

                if num_octave_levels > 0:

                    im_sigma_octave_max_rszd = resize(
                        im_sigma_octave_max, im_dog_max.shape, order=0)

                else:

                    im_sigma_octave_max_rszd = im_sigma_octave_max

                im_sigma_max[max_pixels] = \
                    im_sigma_octave_max_rszd[max_pixels]

            # downsample images by 2 at the end of each octave
            if n_level == num_octave_levels:

                im_gauss_cur = im_gauss_next[::2, ::2]
                im_sigma_ubound_cur = im_sigma_ubound_cur[::2, ::2]

                im_dog_octave_max = im_dog_octave_max[::2, ::2]
                im_sigma_octave_max = im_sigma_octave_max[::2, ::2]

                n_level = 0
                n_octave += 1

    # set min vals to min response
    im_dog_max[im_dog_max == MIN_FLOAT] = 0

    return im_dog_max, im_sigma_max
