import numpy as np

import histomicstk as htk
import histomicstk.filters.shape as htk_shape_filters


def detect_nuclei_kofahi(im_nuclei_stain, im_nuclei_fgnd_mask, min_radius,
                         max_radius, min_nucleus_area, local_max_search_radius):
    """Performs a nuclear segmentation using kofahi's method.

    This method uses scale-adaptive multi-scale Laplacian-of-Gaussian filtering
    for blob enhancement and a local maximum clustering for segmentation. The
    original implementation described by Al-Kofahi et al. uses Laplacian of Gaussian
    but this function replaces it with Difference of Gaussian to improve speed.

    Parameters
    ----------
    im_nuclei_stain : array_like
        A hematoxylin intensity image obtained from ColorDeconvolution.
    im_nuclei_fgnd_mask: array_like
        A binary mask of the nuclear foreground typically obtained by applying
        a threshold on the hematoxylin/nuclei stain image
    min_radius : float
        Minimum nuclear radius (used to set min sigma of the multiscale LoG filter)
    max_radius : float
        Maximum nuclear radius (used to set max sigma of the multiscale LoG filter)
    min_nucleus_area : int
        Minimum area that each nucleus should have
    local_max_search_radius : float
        Local max search radius used for detection seed points in nuclei

    Returns
    -------
    im_nuclei_seg_mask : array_like
        A 2D array mask of the nuclei segmentation.

    References
    ----------
    .. [#] Y. Al-Kofahi et al "Improved Automatic Detection and Segmentation of
       Cell Nuclei in Histopathology Images" in IEEE Transactions on Biomedical
       Engineering, Volume: 57, Issue: 4, doi: 10.1109/TBME.2009.2035102,
       April 2010.

    """
    import scipy as sp
    import skimage.morphology

    # smooth foreground mask with closing and opening
    im_nuclei_fgnd_mask = skimage.morphology.closing(
        im_nuclei_fgnd_mask, skimage.morphology.disk(3))

    im_nuclei_fgnd_mask = skimage.morphology.opening(
        im_nuclei_fgnd_mask, skimage.morphology.disk(3))

    im_nuclei_fgnd_mask = sp.ndimage.binary_fill_holes(
        im_nuclei_fgnd_mask)

    if not np.any(im_nuclei_fgnd_mask):
        return im_nuclei_fgnd_mask

    # run adaptive multi-scale LoG filter
    im_log_max, im_sigma_max = htk_shape_filters.cdog(
        im_nuclei_stain, im_nuclei_fgnd_mask,
        sigma_min=min_radius / np.sqrt(2),
        sigma_max=max_radius / np.sqrt(2)
    )

    # apply local maximum clustering
    im_nuclei_seg_mask, seeds, maxima = htk.segmentation.nuclear.max_clustering(
        im_log_max, im_nuclei_fgnd_mask, local_max_search_radius)

    if seeds is None:
        return im_nuclei_seg_mask

    # split any objects with disconnected fragments
    im_nuclei_seg_mask = htk.segmentation.label.split(im_nuclei_seg_mask,
                                                      conn=8)

    # filter out small objects
    im_nuclei_seg_mask = htk.segmentation.label.area_open(
        im_nuclei_seg_mask, min_nucleus_area).astype(int)

    return im_nuclei_seg_mask
