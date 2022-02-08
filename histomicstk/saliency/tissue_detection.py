#!/usr/bin/env python3
"""
Created on Wed Sep 18 03:29:24 2019.

@author: mtageld
"""

import cv2
import numpy as np
from PIL import Image

from histomicstk.annotations_and_masks.annotation_and_mask_utils import \
    get_image_from_htk_response
from histomicstk.annotations_and_masks.masks_to_annotations_handler import (
    get_annotation_documents_from_contours, get_contours_from_mask)
from histomicstk.preprocessing.color_deconvolution.color_deconvolution import \
    color_deconvolution_routine

Image.MAX_IMAGE_PIXELS = None

# %%===========================================================================


def get_slide_thumbnail(gc, slide_id):
    """Get slide thumbnail using girder client.

    Parameters
    -------------
    gc : object
        girder client to use
    slide_id : str
        girder ID of slide

    Returns
    ---------
    np array
        RGB slide thumbnail at lowest level

    """
    getStr = "/item/%s/tiles/thumbnail" % (slide_id)
    resp = gc.get(getStr, jsonResp=False)
    return get_image_from_htk_response(resp)

# %%===========================================================================


def _deconv_color(im, **kwargs):
    """Wrap around color_deconvolution_routine (compatibility)."""
    Stains, _, _ = color_deconvolution_routine(im, **kwargs)
    return Stains, 0

# %%===========================================================================


def get_tissue_mask(
        thumbnail_im,
        deconvolve_first=False, stain_unmixing_routine_kwargs=None,
        n_thresholding_steps=1, sigma=0., min_size=500):
    """Get binary tissue mask from slide thumbnail.

    Parameters
    -----------
    thumbnail_im : np array
        (m, n, 3) nd array of thumbnail RGB image
        or (m, n) nd array of thumbnail grayscale image
    deconvolve_first : bool
        use hematoxylin channel to find cellular areas?
        This will make things ever-so-slightly slower but is better in
        getting rid of sharpie marker (if it's green, for example).
        Sometimes things work better without it, though.
    stain_matrix_method : str
        see deconv_color method in seed_utils
    n_thresholding_steps : int
        number of gaussian smoothign steps
    sigma : float
        sigma of gaussian filter
    min_size : int
        minimum size (in pixels) of contiguous tissue regions to keep

    Returns
    --------
    np int32 array
        each unique value represents a unique tissue region
    np bool array
        largest contiguous tissue region.

    """
    from scipy import ndimage
    from skimage.filters import gaussian, threshold_otsu

    stain_unmixing_routine_kwargs = (
        {} if stain_unmixing_routine_kwargs is None else stain_unmixing_routine_kwargs)

    if deconvolve_first and (len(thumbnail_im.shape) == 3):
        # deconvolvve to ge hematoxylin channel (cellular areas)
        # hematoxylin channel return shows MINIMA so we invert
        stain_unmixing_routine_kwargs['stains'] = ['hematoxylin', 'eosin']
        Stains, _, _ = color_deconvolution_routine(
            thumbnail_im, **stain_unmixing_routine_kwargs)
        thumbnail = 255 - Stains[..., 0]

    elif len(thumbnail_im.shape) == 3:
        # grayscale thumbnail (inverted)
        thumbnail = 255 - cv2.cvtColor(thumbnail_im, cv2.COLOR_BGR2GRAY)

    else:
        thumbnail = thumbnail_im

    for _ in range(n_thresholding_steps):

        # gaussian smoothing of grayscale thumbnail
        if sigma > 0.0:
            thumbnail = gaussian(
                thumbnail, sigma=sigma,
                output=None, mode='nearest', preserve_range=True)

        # get threshold to keep analysis region
        try:
            thresh = threshold_otsu(thumbnail[thumbnail > 0])
        except ValueError:  # all values are zero
            thresh = 0

        # replace pixels outside analysis region with upper quantile pixels
        thumbnail[thumbnail < thresh] = 0

    # convert to binary
    mask = 0 + (thumbnail > 0)

    # find connected components
    labeled, _ = ndimage.label(mask)

    # only keep
    unique, counts = np.unique(labeled[labeled > 0], return_counts=True)
    discard = np.in1d(labeled, unique[counts < min_size])
    discard = discard.reshape(labeled.shape)
    labeled[discard] = 0

    # largest tissue region
    mask = labeled == unique[np.argmax(counts)]

    return labeled, mask


# %%===========================================================================

def get_tissue_boundary_annotation_documents(
        gc, slide_id, labeled,
        color='rgb(0,0,0)', group='tissue', annprops=None):
    """Get annotation documents of tissue boundaries to visualize on DSA.

    Parameters
    -----------
    gc : object
        girder client to use
    slide_id : str
        girder ID of slide
    labeled : np array
        mask of tissue regions using slide thumbnail. This could either be
        a binary mask or a mask where each unique value corresponds to one
        tissue region. It will be binalized anyways. This can be obtained
        using get_tissue_mask().
    color : str
        color to assign to boundaries. format like rgb(0,0,0)
    group : str
        label for annotations
    annpops : dict
        properties of annotation elements. Contains the following keys
        F, X_OFFSET, Y_OFFSET, opacity, lineWidth. Refer to
        get_single_annotation_document_from_contours() for details.

    Returns
    --------
    list of dicts
        each dict is an annotation document that you can post to DSA

    """
    from pandas import DataFrame

    # Get annotations properties
    if annprops is None:
        slide_info = gc.get('item/%s/tiles' % slide_id)
        annprops = {
            'F': slide_info['sizeX'] / labeled.shape[1],  # relative to base
            'X_OFFSET': 0,
            'Y_OFFSET': 0,
            'opacity': 0,
            'lineWidth': 4.0,
        }

    # Define GTCodes dataframe
    GTCodes_df = DataFrame(columns=['group', 'GT_code', 'color'])
    GTCodes_df.loc['tissue', 'group'] = group
    GTCodes_df.loc['tissue', 'GT_code'] = 1
    GTCodes_df.loc['tissue', 'color'] = color

    # get annotation docs
    contours_tissue = get_contours_from_mask(
        MASK=0 + (labeled > 0), GTCodes_df=GTCodes_df,
        get_roi_contour=False, MIN_SIZE=0, MAX_SIZE=None, verbose=False,
        monitorPrefix="tissue: getting contours")
    annotation_docs = get_annotation_documents_from_contours(
        contours_tissue.copy(), docnamePrefix='test', annprops=annprops,
        verbose=False, monitorPrefix="tissue : annotation docs")

    return annotation_docs

# %%===========================================================================


def threshold_multichannel(
        im, thresholds, channels=None,
        just_threshold=False, get_tissue_mask_kwargs=None):
    """Threshold a multi-channel image (eg. HSI image) to get tissue.

    The relies on the fact that oftentimes some slide elements (eg blood
    or whitespace) have a characteristic hue/saturation/intensity. This
    thresholds along each HSI channel, then optionally uses the
    get_tissue_mask() method (gaussian smoothing, otsu thresholding,
    connected components) to get each contiguous tissue piece.

    Parameters
    -----------
    im : np array
        (m, n, 3) array of Hue, Saturation, Intensity (in this order)
    thresholds : dict
        Each entry is a dict containing the keys min and max
    channels : list
        names of channels, in order (eg. hue, saturation, intensity)
    just_threshold : bool
        if Fase, get_tissue_mask() is used to smooth result and get regions.
    get_tissue_mask_kwargs : dict
        key-value pairs of parameters to pass to get_tissue_mask()

    Returns
    --------
    np int32 array
        if not just_threshold, unique values represent unique tissue regions
    np bool array
        if not just_threshold, largest contiguous tissue region.

    """
    channels = ['hue', 'saturation', 'intensity'] if channels is None else channels

    if get_tissue_mask_kwargs is None:
        get_tissue_mask_kwargs = {
            'n_thresholding_steps': 1,
            'sigma': 5.0,
            'min_size': 10,
        }

    # threshold each channel
    mask = np.ones(im.shape[:2])
    for ax, ch in enumerate(channels):

        channel = im[..., ax].copy()

        mask[channel < thresholds[ch]['min']] = 0
        mask[channel >= thresholds[ch]['max']] = 0

    # smoothing, otsu thresholding then connected components
    if just_threshold or (np.unique(mask).shape[0] < 1):
        labeled = mask
    else:
        get_tissue_mask_kwargs['deconvolve_first'] = False
        labeled, mask = get_tissue_mask(mask, **get_tissue_mask_kwargs)

    return labeled, mask

# %%===========================================================================


def _get_largest_regions(labeled, top_n=10):

    labeled_im = labeled.copy()

    unique, counts = np.unique(labeled_im[labeled_im > 0], return_counts=True)

    keep = unique[np.argsort(counts)[-top_n:]]

    mask = np.zeros(labeled_im.shape)
    keep_pixels = np.in1d(labeled_im, keep)
    keep_pixels = keep_pixels.reshape(labeled_im.shape)
    mask[keep_pixels] = 1
    labeled_im[mask == 0] = 0

    return labeled_im

# %%===========================================================================
