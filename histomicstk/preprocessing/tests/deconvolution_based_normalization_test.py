#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 01:31:38 2019

@author: mtageld
"""
import unittest
import girder_client
import numpy as np
from skimage.transform import resize
# from matplotlib import pylab as plt
# from matplotlib.colors import ListedColormap
from histomicstk.saliency.tissue_detection import (
    get_slide_thumbnail, get_tissue_mask)
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    get_image_from_htk_response)
from histomicstk.preprocessing.color_normalization.\
    deconvolution_based_normalization import deconvolution_based_normalization

# %%===========================================================================
# Constants & prep work

APIURL = 'http://candygram.neurology.emory.edu:8080/api/v1/'
SAMPLE_SLIDE_ID = "5d817f5abd4404c6b1f744bb"

gc = girder_client.GirderClient(apiUrl=APIURL)
# gc.authenticate(interactive=True)
gc.authenticate(apiKey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')

MAG = 1.0

# TCGA-A2-A3XS-DX1_xmin21421_ymin37486_.png, Amgad et al, 2019)
# for macenco (obtained using rgb_separate_stains_macenko_pca()
# and using reordered such that columns are the order:
# Hamtoxylin, Eosin, Null
W_target = np.array([
    [0.5807549,  0.08314027,  0.08213795],
    [0.71681094,  0.90081588,  0.41999816],
    [0.38588316,  0.42616716, -0.90380025]
])

# %%===========================================================================

print("Getting images to be normalized ...")

# get RGB image at a small magnification
slide_info = gc.get('item/%s/tiles' % SAMPLE_SLIDE_ID)
getStr = "/item/%s/tiles/region?left=%d&right=%d&top=%d&bottom=%d" % (
    SAMPLE_SLIDE_ID, 0, slide_info['sizeX'], 0, slide_info['sizeY']
    ) + "&magnification=%.2f" % MAG
tissue_rgb = get_image_from_htk_response(
    gc.get(getStr, jsonResp=False))

# get mask of things to ignore
thumbnail_rgb = get_slide_thumbnail(gc, SAMPLE_SLIDE_ID)
mask_out, _ = get_tissue_mask(
    thumbnail_rgb, deconvolve_first=True,
    n_thresholding_steps=1, sigma=1.5, min_size=30)
mask_out = resize(
    mask_out == 0, output_shape=tissue_rgb.shape[:2],
    order=0, preserve_range=True) == 1

# %%===========================================================================


class DeconvolutionBasedNormalizationTest(unittest.TestCase):
    """Test deconvolution normalization."""

    def test_macenko_normalization(self):
        """Test macenko_pca normalization."""
        print("Macenko - Unmasked, using default, 'idealized' W_target")
        tissue_rgb_normalized = deconvolution_based_normalization(
            tissue_rgb, stain_deconvolution_method='macenko_pca')
        self.assertTupleEqual(tuple(
            [int(tissue_rgb_normalized[..., i].mean()) for i in range(3)]),
            (209, 186, 232)
        )

        print("Macenko - Unmasked, using W_target from good image")
        tissue_rgb_normalized = deconvolution_based_normalization(
            tissue_rgb, W_target=W_target,
            stain_deconvolution_method='macenko_pca')
        self.assertTupleEqual(tuple(
            [int(tissue_rgb_normalized[..., i].mean()) for i in range(3)]),
            (213, 189, 212)
        )

        print("Macenko - Masked, using W_target from good image")
        tissue_rgb_normalized = deconvolution_based_normalization(
            tissue_rgb, W_target=W_target,
            stain_deconvolution_method='macenko_pca', mask_out=mask_out)
        self.assertTupleEqual(tuple(
            [int(tissue_rgb_normalized[..., i].mean()) for i in range(3)]),
            (212, 195, 215)
        )

    # def test_xu_normalization(self):
    #     """Test xu_snmf normalization. (VERY SLOW!!)"""
    #     # Unmasked using W_target from good image
    #     tissue_rgb_normalized = deconvolution_based_normalization(
    #         tissue_rgb, stain_deconvolution_method='xu_snmf')


# %%===========================================================================


if __name__ == '__main__':
    unittest.main()


# %%
from histomicstk.preprocessing.color_deconvolution.\
    rgb_separate_stains_macenko_pca import rgb_separate_stains_macenko_pca
from histomicstk.preprocessing.color_deconvolution.color_deconvolution import (
    color_deconvolution)
from histomicstk.preprocessing import color_conversion
import histomicstk.utils as utils

# get W_source
W_source = rgb_separate_stains_macenko_pca(
    tissue_rgb, I_0=None, mask_out=mask_out)

# find stains matrix from source image
_, StainsFloat, _ = color_deconvolution(tissue_rgb, w=W_source, I_0=None)

# %%


def augment_stain_concentration(
        StainsFloat, W, sda_fwd=None, I_0=None, mask_out=None):

    # augment everything, otherwise only augment specific pixels
    if mask_out is None:
        keep_mask = np.zeros(StainsFloat.shape[:2]) == 0
    else:
        keep_mask = np.equal(mask_out, False)
    keep_mask = np.tile(keep_mask[..., None], (1, 1, 3))
    keep_mask = utils.convert_image_to_matrix(keep_mask)

    # transform 3D input stain image to 2D stain matrix format
    m = utils.convert_image_to_matrix(StainsFloat)

    # transform input stains to optical density values
    if sda_fwd is None:
        sda_fwd = color_conversion.rgb_to_sda(
            m, 255 if I_0 is not None else None, allow_negatives=True)

    # perturb concentrations in SDA space
    augmented_sda = sda_fwd.copy()

    sigma1 = 0.8
    sigma2 = 0.8

    for i in range(3):
        alpha = np.random.uniform(1 - sigma1, 1 + sigma1)
        beta = np.random.uniform(-sigma2, sigma2)
        augmented_sda[i, keep_mask[i, :]] *= alpha
        augmented_sda[i, keep_mask[i, :]] += beta

    sda_conv = np.dot(W, augmented_sda)
    sda_inv = color_conversion.sda_to_rgb(sda_conv, I_0)

    # reshape output, transform type
    augmented_rgb = (
        utils.convert_matrix_to_image(sda_inv, StainsFloat.shape)
        .clip(0, 255).astype(np.uint8))

    return augmented_rgb

# %%

sda_fwd = color_conversion.rgb_to_sda(
            m, 255 if I_0 is not None else None, allow_negatives=True)
augmented_rgb = augment_stain_concentration()
plt.figure(figsize=(7,7))
plt.imshow(augmented_rgb[1300:1500, 1300:1500, :])




