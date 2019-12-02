#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 00:06:28 2019.

@author: mtageld
"""

import unittest

import os
import tempfile
import shutil
from imageio import imread, imwrite
import girder_client
import numpy as np
# from matplotlib import pylab as plt
# from matplotlib.colors import ListedColormap
from histomicstk.saliency.tissue_detection import (
    get_slide_thumbnail, get_tissue_mask,
    get_tissue_boundary_annotation_documents)
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    delete_annotations_in_slide)

# %%===========================================================================
# Constants & prep work

APIURL = 'http://candygram.neurology.emory.edu:8080/api/v1/'
SAMPLE_SLIDE_ID = "5d586d76bd4404c6b1f286ae"

gc = girder_client.GirderClient(apiUrl=APIURL)
# gc.authenticate(interactive=True)
gc.authenticate(apiKey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')

savepath = tempfile.mkdtemp()

# %%===========================================================================
# Tests


class TissueDetectionTest(unittest.TestCase):
    """Test methods for getting ROI mask from annotations."""

    def test_get_tissue_mask(self):
        """Test get_tissue_mask()."""
        thumbnail_rgb = get_slide_thumbnail(gc, SAMPLE_SLIDE_ID)

        labeled, mask = get_tissue_mask(
            thumbnail_rgb, deconvolve_first=True,
            n_thresholding_steps=1, sigma=1.5, min_size=30)

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

        self.assertTupleEqual(labeled.shape, (156, 256))
        self.assertEqual(len(np.unique(labeled)), 10)

        # save for use in the next test
        imwrite(os.path.join(
            savepath, 'tissue_binmask.png'), np.uint8(0 + (labeled > 0)))

    def test_get_tissue_boundary_annotation_documents(self):
        """Test get_tissue_boundary_annotation_documents()."""
        labeled = imread(os.path.join(savepath, 'tissue_binmask.png'))
        annotation_docs = get_tissue_boundary_annotation_documents(
            gc, slide_id=SAMPLE_SLIDE_ID, labeled=labeled)

        self.assertTrue('elements' in annotation_docs[0].keys())
        self.assertEqual(len(annotation_docs[0]['elements']), 9)

        # deleting existing annotations in target slide (if any)
        delete_annotations_in_slide(gc, SAMPLE_SLIDE_ID)

        # post annotations to slide
        for doc in annotation_docs:
            _ = gc.post("/annotation?itemId=" + SAMPLE_SLIDE_ID, json=doc)


def suite():
    """Run chained unit tests in desired order.

    See: https://stackoverflow.com/questions/5387299/...
         ... python-unittest-testcase-execution-order
    """
    suite = unittest.TestSuite()
    suite.addTest(TissueDetectionTest('test_get_tissue_mask'))
    suite.addTest(
        TissueDetectionTest('test_get_tissue_boundary_annotation_documents'))
    return suite

# %%===========================================================================


if __name__ == '__main__':
    runner = unittest.TextTestRunner(failfast=True)
    runner.run(suite())
    # cleanup
    shutil.rmtree(savepath)
