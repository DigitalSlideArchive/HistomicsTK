#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 01:31:38 2019.

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

# since this is a unit test, just work on a small image
tissue_rgb = tissue_rgb[1000:1500, 2500:3000, :]
mask_out = mask_out[1000:1500, 2500:3000]

# %%===========================================================================


class DeconvolutionBasedNormalizationTest(unittest.TestCase):
    """Test deconvolution normalization."""

    def test_macenko_normalization(self):
        """Test macenko_pca normalization."""
        stain_unmixing_routine_params = {
            'stains': ['hematoxylin', 'eosin'],
            'stain_unmixing_method': 'macenko_pca',
        }

        print("Macenko - Unmasked, using default, 'idealized' W_target")
        tissue_rgb_normalized = deconvolution_based_normalization(
            tissue_rgb,
            stain_unmixing_routine_params=stain_unmixing_routine_params)
        self.assertTupleEqual(tuple(
            [int(tissue_rgb_normalized[..., i].mean()) for i in range(3)]),
            (192, 161, 222)
        )

        print("Macenko - Unmasked, using W_target from good image")
        tissue_rgb_normalized = deconvolution_based_normalization(
            tissue_rgb, W_target=W_target,
            stain_unmixing_routine_params=stain_unmixing_routine_params)
        self.assertTupleEqual(tuple(
            [int(tissue_rgb_normalized[..., i].mean()) for i in range(3)]),
            (198, 163, 197)
        )

        print("Macenko - Masked, using W_target from good image")
        tissue_rgb_normalized = deconvolution_based_normalization(
            tissue_rgb, W_target=W_target, mask_out=mask_out,
            stain_unmixing_routine_params=stain_unmixing_routine_params)
        self.assertTupleEqual(tuple(
            [int(tissue_rgb_normalized[..., i].mean()) for i in range(3)]),
            (194, 172, 201)
        )

    # def test_xu_normalization(self):
    #     """Test xu_snmf normalization. (VERY SLOW!!)"""
    #     stain_unmixing_routine_params = {
    #        'stains': ['hematoxylin', 'eosin'],
    #        'stain_unmixing_method': 'xu_snmf',
    #     }
    #     # Unmasked using W_target from good image
    #     tissue_rgb_normalized = deconvolution_based_normalization(
    #         tissue_rgb,
    #         stain_unmixing_routine_params=stain_unmixing_routine_params)


# %%===========================================================================


if __name__ == '__main__':
    unittest.main()
