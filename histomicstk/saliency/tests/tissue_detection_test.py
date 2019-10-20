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
    get_tissue_boundary_annotation_documents, threshold_hsi)
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    delete_annotations_in_slide, get_image_from_htk_response)
from histomicstk.preprocessing.color_conversion import rgb_to_hsi

# %%===========================================================================
# Constants & prep work

APIURL = 'http://candygram.neurology.emory.edu:8080/api/v1/'
# SAMPLE_SLIDE_ID = "5d586d76bd4404c6b1f286ae"
SAMPLE_SLIDE_ID = "5d94ee48bd4404c6b1fb0b40"

gc = girder_client.GirderClient(apiUrl=APIURL)
# gc.authenticate(interactive=True)
gc.authenticate(apiKey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')

savepath = tempfile.mkdtemp()

whitespace_hsi_thresholds = {
    'hue': {'min': 0, 'max': 1.0},
    'saturation': {'min': 0, 'max': 0.2},
    'intensity': {'min': 220, 'max': 255},
}

blood_hsi_thresholds = {
    'hue': {'min': 0.15, 'max': 0.2},
    'saturation': {'min': 0.55, 'max': 1.0},
    'intensity': {'min': 0, 'max': 255},
}

necrosis_hsi_thresholds = {
    'hue': {'min': 0.15, 'max': 0.2},
    'saturation': {'min': 0.35, 'max': 0.55},
    'intensity': {'min': 0, 'max': 255},
}

# %%===========================================================================
# Tests


class TissueDetectionTest(unittest.TestCase):
    """Test methods for detecting tissue."""

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

    def test_threshold_hsi(self):
        """Test threshold_hsi()."""

        # load RGB for ROI at target magnification
        getStr = "/item/%s/tiles/region?left=%d&right=%d&top=%d&bottom=%d" % (
            SAMPLE_SLIDE_ID, 22414, 84140, 13837, 67967
            ) + "&magnification=%d" % 3.0
        resp = gc.get(getStr, jsonResp=False)
        tissue_im = get_image_from_htk_response(resp)

        # convert to hsi
        tissue_im = rgb_to_hsi(tissue_im)

        # extract specific regions
        labeled, mask = threshold_hsi(
            tissue_im,
            # hsi_thresholds=whitespace_hsi_thresholds,  # whitespace
            hsi_thresholds=blood_hsi_thresholds,  # blood
            # hsi_thresholds=necrosis_hsi_thresholds,  # necrosis
            just_threshold=False, get_tissue_mask_kwargs={
                'n_thresholding_steps': 1, 'sigma': 5.0, 'min_size': 100},
        )

        self.assertTupleEqual(labeled.shape, (4059, 4629))
        # self.assertEqual(len(np.unique(labeled)), 659)

        # save for use in the next test
        imwrite(os.path.join(
            savepath, 'region_binmask.png'), np.uint8(0 + (labeled > 0)))

    def visualize_threshold_hsi_annotations(self):
        """Visualize results from threshold_hsi()."""
        labeled = imread(os.path.join(savepath, 'region_binmask.png'))

        # deleting existing annotations in target slide (if any)
        # delete_annotations_in_slide(gc, SAMPLE_SLIDE_ID)

        # get annotation documents
        slide_info = gc.get('item/%s/tiles' % SAMPLE_SLIDE_ID)
        annotation_docs = get_tissue_boundary_annotation_documents(
            gc, slide_id=SAMPLE_SLIDE_ID, labeled=labeled,
            group='blood', color='rgb(255,255,0)', docnamePrefix='test',
            # group='whitespace', color='rgb(70,70,70)', docnamePrefix='test',
            # group='necrosis', color='rgb(255,180,70)', docnamePrefix='test',
            annprops={
                'F': slide_info['magnification'] / 3.0,
                'X_OFFSET': 22414,
                'Y_OFFSET': 13837,
                'opacity': 0,
                'lineWidth': 4.0,
            }, )

        self.assertTrue('elements' in annotation_docs[0].keys())

        # post annotations to slide
        for doc in annotation_docs:
            _ = gc.post("/annotation?itemId=" + SAMPLE_SLIDE_ID, json=doc)


def suite():
    """Run chained unit tests in desired order.

    See: https://stackoverflow.com/questions/5387299/...
         ... python-unittest-testcase-execution-order
    """
    suite = unittest.TestSuite()
    # suite.addTest(TissueDetectionTest('test_get_tissue_mask'))
    # suite.addTest(
    #     TissueDetectionTest('test_get_tissue_boundary_annotation_documents'))
    suite.addTest(TissueDetectionTest('test_threshold_hsi'))
    suite.addTest(
        TissueDetectionTest('visualize_threshold_hsi_annotations'))
    return suite

# %%===========================================================================


if __name__ == '__main__':
    runner = unittest.TextTestRunner(failfast=True)
    runner.run(suite())

    # cleanup
    shutil.rmtree(savepath)
