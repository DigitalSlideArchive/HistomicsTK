#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 00:06:28 2019.

@author: mtageld
"""
import pytest
import os
import numpy as np
# from matplotlib import pylab as plt
from histomicstk.saliency.tissue_detection import (
    get_slide_thumbnail, get_tissue_mask,
    get_tissue_boundary_annotation_documents)
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    delete_annotations_in_slide)
import sys
thisDir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(thisDir, '../../../tests'))
# import htk_test_utilities as utilities  # noqa
from htk_test_utilities import girderClient, getTestFilePath  # noqa
# # for protyping
# from tests.htk_test_utilities import _connect_to_existing_local_dsa
# girderClient = _connect_to_existing_local_dsa()


class Cfg:
    def __init__(self):
        self.gc = None
        self.iteminfo = None
        self.labeled = None


cfg = Cfg()


class TestTissueDetection():
    """Test methods for getting ROI mask from annotations."""

    @pytest.mark.usefixtures('girderClient')  # noqa
    def test_prep(self, girderClient):  # noqa
        cfg.gc = girderClient
        cfg.iteminfo = cfg.gc.get('/item', parameters={
            'text': "TCGA-A2-A0YE-01Z-00-DX1"})[0]

    def test_get_tissue_mask(self):
        """Test get_tissue_mask()."""
        thumbnail_rgb = get_slide_thumbnail(cfg.gc, cfg.iteminfo['_id'])
        cfg.labeled, mask = get_tissue_mask(
            thumbnail_rgb, deconvolve_first=True,
            n_thresholding_steps=1, sigma=1.5, min_size=30)

        assert cfg.labeled.shape == (156, 256)
        assert len(np.unique(cfg.labeled)) == 11

    def test_get_tissue_boundary_annotation_documents(self):
        """Test get_tissue_boundary_annotation_documents()."""
        annotation_docs = get_tissue_boundary_annotation_documents(
            cfg.gc, slide_id=cfg.iteminfo['_id'], labeled=cfg.labeled)

        assert 'elements' in annotation_docs[0].keys()
        assert len(annotation_docs[0]['elements']) == 10

        # test deleting existing annotations in slide
        delete_annotations_in_slide(cfg.gc, cfg.iteminfo['_id'])

        # check that it posts without issues
        resps = [
            cfg.gc.post(
                "/annotation?itemId=" + cfg.iteminfo['_id'], json=doc)
            for doc in annotation_docs
        ]
        assert all(['annotation' in resp for resp in resps])
