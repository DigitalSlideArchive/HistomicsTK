#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 00:06:28 2019.

@author: mtageld
"""

import unittest

import girder_client
import numpy as np
# from matplotlib import pylab as plt
# from matplotlib.colors import ListedColormap
from histomicstk.saliency.tissue_detection import (
    get_slide_thumbnail, get_tissue_mask)

# %%===========================================================================
# Constants & prep work

APIURL = 'http://candygram.neurology.emory.edu:8080/api/v1/'
# SAMPLE_SLIDE_ID = '5d586d57bd4404c6b1f28640'
SAMPLE_SLIDE_ID = "5d817f5abd4404c6b1f744bb"

gc = girder_client.GirderClient(apiUrl=APIURL)
# gc.authenticate(interactive=True)
gc.authenticate(apiKey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')


# %%===========================================================================
# Tests
# =============================================================================

class TissueDetectionTest(unittest.TestCase):
    """Test methods for getting ROI mask from annotations."""

    def test_get_tissue_mask(self):
        """Test get_tissue_mask()."""
        thumbnail_rgb = get_slide_thumbnail(gc, SAMPLE_SLIDE_ID)

        labeled, mask = get_tissue_mask(
            thumbnail_rgb, deconvolve_first=True,
            n_thresholding_steps=2, sigma=0., min_size=30)

        # # visualize result
        # vals = np.random.rand(256,3)
        # vals[0, ...] = [0.9, 0.9, 0.9]
        # cMap = ListedColormap(1 - vals)
        #
        # f, ax = plt.subplots(1, 3, figsize=(20, 20))
        # ax[0].imshow(thumbnail_rgb)
        # ax[1].imshow(labeled, cmap=cMap)
        # ax[2].imshow(mask, cmap=cMap)
        # plt.show()

        self.assertTupleEqual(labeled.shape, (152, 256))
        self.assertEqual(len(np.unique(labeled)), 10)


# %%===========================================================================


if __name__ == '__main__':
    unittest.main()
