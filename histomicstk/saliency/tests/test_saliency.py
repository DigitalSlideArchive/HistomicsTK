"""
Created on Wed Sep 18 00:06:28 2019.

@author: mtageld
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import pytest
from pandas import read_csv

from histomicstk.annotations_and_masks.annotation_and_mask_utils import \
    delete_annotations_in_slide
from histomicstk.saliency.cellularity_detection_superpixels import \
    Cellularity_detector_superpixels
from histomicstk.saliency.cellularity_detection_thresholding import \
    Cellularity_detector_thresholding
from histomicstk.saliency.tissue_detection import (
    get_slide_thumbnail, get_tissue_boundary_annotation_documents,
    get_tissue_mask)

thisDir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(thisDir, '../../../tests'))
# import htk_test_utilities as utilities  # noqa
from htk_test_utilities import getTestFilePath, girderClient  # noqa


class Cfg:
    def __init__(self):
        self.gc = None
        self.iteminfo = None
        self.labeled = None
        self.GTcodes = None
        self.logging_savepath_cdt = tempfile.mkdtemp()
        self.logging_savepath_cds = tempfile.mkdtemp()


cfg = Cfg()


@pytest.mark.usefixtures('girderClient')  # noqa
def test_prep(girderClient):  # noqa
    cfg.gc = girderClient
    cfg.iteminfo = cfg.gc.get('/item', parameters={'text': 'TCGA-A2-A0YE-01Z-00-DX1'})[0]
    cfg.GTcodes = read_csv(getTestFilePath('saliency_GTcodes.csv'))


class TestTissueDetection():
    """Test methods for getting ROI mask from annotations."""

    def test_get_tissue_mask(self):
        """Test get_tissue_mask()."""
        thumbnail_rgb = get_slide_thumbnail(cfg.gc, cfg.iteminfo['_id'])
        cfg.labeled, mask = get_tissue_mask(
            thumbnail_rgb, deconvolve_first=True,
            n_thresholding_steps=1, sigma=1.5, min_size=30)

        assert cfg.labeled.shape == (156, 256)
        assert len(np.unique(cfg.labeled)) in (10, 11)

    def test_get_tissue_boundary_annotation_documents(self):
        """Test get_tissue_boundary_annotation_documents()."""
        annotation_docs = get_tissue_boundary_annotation_documents(
            cfg.gc, slide_id=cfg.iteminfo['_id'], labeled=cfg.labeled)

        assert 'elements' in annotation_docs[0].keys()
        assert len(annotation_docs[0]['elements']) in {9, 10}

        # test delete existing annotations in slide
        delete_annotations_in_slide(cfg.gc, cfg.iteminfo['_id'])

        # check that it posts without issues
        resps = [
            cfg.gc.post(
                '/annotation?itemId=' + cfg.iteminfo['_id'], json=doc)
            for doc in annotation_docs
        ]
        assert all('annotation' in resp for resp in resps)


class TestCellularityDetection:
    """Test methods for getting cellularity."""

    def test_cellularity_detection_thresholding(self):
        """Test Cellularity_detector_thresholding()."""
        # run cellularity detector
        cdt = Cellularity_detector_thresholding(
            cfg.gc, slide_id=cfg.iteminfo['_id'],
            GTcodes=cfg.GTcodes, MAG=1.0,
            verbose=1, monitorPrefix='test',
            logging_savepath=cfg.logging_savepath_cdt)
        tissue_pieces = cdt.run()

        # check
        assert len(tissue_pieces) == 3
        assert all(j in tissue_pieces[0].__dict__.keys() for j in
                   ('labeled', 'ymin', 'xmin', 'ymax', 'xmax'))

        # cleanup
        shutil.rmtree(cfg.logging_savepath_cdt)

    def test_cellularity_detection_superpixels(self):
        """Test Cellularity_detector_superpixels()."""
        # from the ROI in Amgad et al, 2019
        cnorm_main = {
            'mu': np.array([8.74108109, -0.12440419, 0.0444982]),
            'sigma': np.array([0.6135447, 0.10989545, 0.0286032]),
        }
        # run cellularity detector
        cds = Cellularity_detector_superpixels(
            cfg.gc, slide_id=cfg.iteminfo['_id'],
            MAG=1.0, compactness=0.1, spixel_size_baseMag=256 * 256,
            max_cellularity=40,
            visualize_spixels=True, visualize_contiguous=True,
            get_tissue_mask_kwargs={
                'deconvolve_first': False,
                'n_thresholding_steps': 2,
                'sigma': 1.5,
                'min_size': 500},
            verbose=1, monitorPrefix='test',
            logging_savepath=cfg.logging_savepath_cds)
        cds.set_color_normalization_values(
            mu=cnorm_main['mu'], sigma=cnorm_main['sigma'], what='main')
        tissue_pieces = cds.run()

        # check
        assert len(tissue_pieces) == 2
        assert all(j in tissue_pieces[0].__dict__.keys()
                   for j in ('tissue_mask', 'ymin', 'xmin', 'ymax', 'xmax',
                             'spixel_mask', 'fdata', 'cluster_props'))
        assert len(tissue_pieces[0].cluster_props) == 5

        # cleanup
        shutil.rmtree(cfg.logging_savepath_cds)
