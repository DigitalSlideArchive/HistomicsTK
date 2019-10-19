#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 00:58:54 2019.

@author: mtageld
"""
import unittest
import girder_client
import numpy as np
from skimage.transform import resize
# from matplotlib import pylab as plt
# from matplotlib.colors import ListedColormap
from histomicstk.preprocessing.color_conversion import lab_mean_std
from histomicstk.preprocessing.color_normalization import reinhard
from histomicstk.saliency.tissue_detection import (
    get_slide_thumbnail, get_tissue_mask)
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    get_image_from_htk_response)

# %%===========================================================================
# Constants & prep work

APIURL = 'http://candygram.neurology.emory.edu:8080/api/v1/'
SAMPLE_SLIDE_ID = "5d817f5abd4404c6b1f744bb"

gc = girder_client.GirderClient(apiUrl=APIURL)
# gc.authenticate(interactive=True)
gc.authenticate(apiKey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')

MAG = 1.0

# color norm. standard (from TCGA-A2-A3XS-DX1, Amgad et al, 2019)
cnorm = {
    'mu': np.array([8.74108109, -0.12440419,  0.0444982]),
    'sigma': np.array([0.6135447, 0.10989545, 0.0286032]),
}

# %%===========================================================================


class ReinhardTest(unittest.TestCase):
    """Test reinhard normalization."""

    def test_reinhard(self):
        """Test reinhard."""
        # get RGB image at a small magnification
        slide_info = gc.get('item/%s/tiles' % SAMPLE_SLIDE_ID)
        getStr = "/item/%s/tiles/region?left=%d&right=%d&top=%d&bottom=%d" % (
            SAMPLE_SLIDE_ID, 0, slide_info['sizeX'], 0, slide_info['sizeY']
            ) + "&magnification=%.2f" % MAG
        tissue_rgb = get_image_from_htk_response(
            gc.get(getStr, jsonResp=False))

        # # SANITY CHECK! normalize to LAB mean and std from SAME slide
        # mean_lab, std_lab = lab_mean_std(tissue_rgb)
        # tissue_rgb_normalized = reinhard(
        #     tissue_rgb, target_mu=mean_lab, target_sigma=std_lab)
        #
        # # we expect the images to be (almost) exactly the same
        # assert np.mean(tissue_rgb - tissue_rgb_normalized) < 1

        # Normalize to pre-set color standard
        tissue_rgb_normalized = reinhard(
            tissue_rgb, target_mu=cnorm['mu'], target_sigma=cnorm['sigma'])

        # check that it matches
        mean_lab, std_lab = lab_mean_std(tissue_rgb_normalized)
        self.assertTrue(all(
            np.abs(mean_lab - cnorm['mu']) < [0.1, 0.1, 0.1]))
        self.assertTrue(all(
            np.abs(std_lab - cnorm['sigma']) < [0.1, 0.1, 0.1]))

        # get tissue mask
        thumbnail_rgb = get_slide_thumbnail(gc, SAMPLE_SLIDE_ID)
        labeled, mask = get_tissue_mask(
            thumbnail_rgb, deconvolve_first=True,
            n_thresholding_steps=1, sigma=1.5, min_size=30)

        # # visualize result
        # vals = np.random.rand(256, 3)
        # vals[0, ...] = [0.9, 0.9, 0.9]
        # cMap = ListedColormap(1 - vals)
        #
        # f, ax = plt.subplots(1, 3, figsize=(20, 20))
        # ax[0].imshow(thumbnail_rgb)
        # ax[1].imshow(labeled, cmap=cMap)
        # ax[2].imshow(mask, cmap=cMap)
        # plt.show()

        # Do MASKED normalization to preset standard
        mask_out = resize(
            labeled == 0, output_shape=tissue_rgb.shape[:2],
            order=0, preserve_range=True) == 1
        tissue_rgb_normalized = reinhard(
            tissue_rgb, target_mu=cnorm['mu'], target_sigma=cnorm['sigma'],
            mask_out=mask_out)

        # check that it matches
        mean_lab, std_lab = lab_mean_std(
            tissue_rgb_normalized, mask_out=mask_out)
        self.assertTrue(all(
            np.abs(mean_lab - cnorm['mu']) < [0.1, 0.1, 0.1]))
        self.assertTrue(all(
            np.abs(std_lab - cnorm['sigma']) < [0.1, 0.1, 0.1]))


# %%===========================================================================


if __name__ == '__main__':
    unittest.main()
