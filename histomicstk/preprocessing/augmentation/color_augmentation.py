"""
Created on Sat Oct 19 18:09:17 2019.

@author: mtageld
"""
import numpy as np

from histomicstk.preprocessing.color_conversion import rgb_to_sda, sda_to_rgb
from histomicstk.preprocessing.color_deconvolution import \
    color_deconvolution_routine
from histomicstk.utils import convert_image_to_matrix, convert_matrix_to_image


def perturb_stain_concentration(
        StainsFloat, W, I_0=None, mask_out=None, sigma1=0.9, sigma2=0.9):
    """Perturb stain concentrations in SDA space and return augmented image.

    This is an implementation of the method described in Tellez et
    al, 2018 (see below). The SDA matrix is perturbed by multiplying each
    channel independently by a value chosen from a random uniform distribution
    in the range [1 - sigma1, 1 + sigma1], then add a value chosen from another
    random uniform distribution in the range [-sigma2, sigma2].

    Parameters
    ------------
    StainsFloat : array_like
        An intensity image (m, n, 3) of deconvolved stains that is unbounded,
        suitable for reconstructing color images of deconvolved stains
        with color_convolution.

    W : array_like
        A 3x3 complemented stain matrix.

    I_0 : float or array_like, optional
        A float a 3-vector containing background RGB intensities.
        If unspecified, use the old OD conversion.

    mask_out : array_like, default is None
        if not None, should be (m x n) boolean numpy array.
        This parameter ensures exclusion of non-masked areas from perturbing.
        This is relevant because elements like blood, sharpie marker,
        white space, etc cannot be simply modeled as a mix of two stains.

    sigma1 : float
        parameter, see beginning of this docstring.

    sigma2 : float
        parameter, see beginning of this docstring.

    Returns
    --------
    array_like
        Color augmented RGB image (m x n x 3)

    References
    ----------
    .. [#] Tellez, David, Maschenka Balkenhol, Irene Otte-Höller,
           Rob van de Loo, Rob Vogels, Peter Bult, Carla Wauters et al.
           "Whole-slide mitosis detection in H&E breast histology using PHH3
           as a reference to train distilled stain-invariant convolutional
           networks." IEEE transactions on medical imaging 37, no. 9
           (2018): 2126-2136.
    .. [#] Tellez, David, Geert Litjens, Peter Bandi, Wouter Bulten,
           John-Melle Bokhorst, Francesco Ciompi, and Jeroen van der Laak.
           "Quantifying the effects of data augmentation and stain color
           normalization in convolutional neural networks for computational
           pathology." arXiv preprint arXiv:1902.06543 (2019).
    .. [#] Implementation inspired by Peter Byfield StainTools repository. See
           https://github.com/Peter554/StainTools/blob/master/LICENSE.txt
           for copyright license (MIT license).

    """
    # augment everything, otherwise only augment specific pixels
    if mask_out is None:
        keep_mask = np.zeros(StainsFloat.shape[:2]) == 0
    else:
        keep_mask = np.equal(mask_out, False)
    keep_mask = np.tile(keep_mask[..., None], (1, 1, 3))
    keep_mask = convert_image_to_matrix(keep_mask)

    # transform 3D input stain image to 2D stain matrix format
    m = convert_image_to_matrix(StainsFloat)

    # transform input stains to optical density values
    sda_fwd = rgb_to_sda(
        m, 255 if I_0 is not None else None, allow_negatives=True)

    # perturb concentrations in SDA space
    augmented_sda = sda_fwd.copy()
    for i in range(3):
        alpha = np.random.uniform(1 - sigma1, 1 + sigma1)
        beta = np.random.uniform(-sigma2, sigma2)
        augmented_sda[i, keep_mask[i, :]] *= alpha
        augmented_sda[i, keep_mask[i, :]] += beta

    # convolve with stains column vectors and convert to RGB
    sda_conv = np.dot(W, augmented_sda)
    sda_inv = sda_to_rgb(sda_conv, I_0)

    # reshape output, transform type
    augmented_rgb = (
        convert_matrix_to_image(sda_inv, StainsFloat.shape)
        .clip(0, 255).astype(np.uint8))

    return augmented_rgb


def rgb_perturb_stain_concentration(
        im_rgb, stain_unmixing_routine_params=None, **kwargs):
    """Apply wrapper that calls perturb_stain_concentration() on RGB.

    Parameters
    ------------
    im_rgb : array_like
        An RGB image (m x n x 3) to color normalize

    stain_unmixing_routine_params : dict
        kwargs to pass as-is to the color_deconvolution_routine().

    kwargs : k,v pairs
        Passed as-is to perturb_stain_concentration()

    Returns
    --------
    array_like
        Color augmented RGB image (m x n x 3)

    """
    stain_unmixing_routine_params = {
        'stains': ['hematoxylin', 'eosin'],
        'stain_unmixing_method': 'macenko_pca',
    } if stain_unmixing_routine_params is None else stain_unmixing_routine_params

    _, StainsFloat, W_source = color_deconvolution_routine(
        im_rgb, W_source=None, **stain_unmixing_routine_params)

    return perturb_stain_concentration(StainsFloat, W_source, **kwargs)
