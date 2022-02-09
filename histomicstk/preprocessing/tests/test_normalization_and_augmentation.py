#!/usr/bin/env python3
"""
Created on Sun Oct 20 00:14:03 2019.

@author: mtageld
"""
import os
import sys

import numpy as np
import pytest
from skimage.transform import resize

from histomicstk.annotations_and_masks.annotation_and_mask_utils import \
    get_image_from_htk_response
from histomicstk.preprocessing.augmentation.color_augmentation import \
    rgb_perturb_stain_concentration
from histomicstk.preprocessing.color_conversion import lab_mean_std
from histomicstk.preprocessing.color_normalization import reinhard
from histomicstk.preprocessing.color_normalization.deconvolution_based_normalization import \
    deconvolution_based_normalization
from histomicstk.saliency.tissue_detection import get_tissue_mask

thisDir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(thisDir, '../../../tests'))
# import htk_test_utilities as utilities  # noqa
from htk_test_utilities import getTestFilePath, girderClient  # noqa

# # for protyping
# from tests.htk_test_utilities import _connect_to_existing_local_dsa
# girderClient = _connect_to_existing_local_dsa()


class Cfg:
    def __init__(self):
        self.gc = None
        self.tissue_rgb = None
        self.mask_out = None


cfg = Cfg()


@pytest.mark.usefixtures('girderClient')  # noqa
def test_prep(girderClient):  # noqa

    cfg.gc = girderClient

    iteminfo = cfg.gc.get('/item', parameters={
        'text': "TCGA-A2-A0YE-01Z-00-DX1"})[0]

    # get RGB region at a small magnification
    MAG = 1.5
    getStr = "/item/%s/tiles/region?left=%d&right=%d&top=%d&bottom=%d&encoding=PNG" % (
        iteminfo['_id'], 46890, 50000, 40350, 43000
        ) + "&magnification=%.2f" % MAG
    cfg.tissue_rgb = get_image_from_htk_response(
        cfg.gc.get(getStr, jsonResp=False))

    # get mask of things to ignore
    cfg.mask_out, _ = get_tissue_mask(
        cfg.tissue_rgb, deconvolve_first=False,
        n_thresholding_steps=1, sigma=1.5, min_size=30)
    cfg.mask_out = resize(
        cfg.mask_out == 0, output_shape=cfg.tissue_rgb.shape[:2],
        order=0, preserve_range=True) == 1


class TestColorNormalization():
    """Test color normalization."""

    def test_reinhard(self):
        """Test reinhard."""
        # # SANITY CHECK! normalize to LAB mean and std from SAME slide
        # mean_lab, std_lab = lab_mean_std(tissue_rgb)
        # tissue_rgb_normalized = reinhard(
        #     tissue_rgb, target_mu=mean_lab, target_sigma=std_lab)
        # # we expect the images to be (almost) exactly the same
        # assert np.mean(tissue_rgb - tissue_rgb_normalized) < 1

        # color norm. standard (from TCGA-A2-A3XS-DX1, Amgad et al, 2019)
        cnorm = {
            'mu': np.array([8.74108109, -0.12440419, 0.0444982]),
            'sigma': np.array([0.6135447, 0.10989545, 0.0286032]),
        }

        # Normalize to pre-set color standard (unmasked)
        tissue_rgb_normalized = reinhard(
            cfg.tissue_rgb, target_mu=cnorm['mu'], target_sigma=cnorm['sigma'])

        # check that it matches
        mean_lab, std_lab = lab_mean_std(tissue_rgb_normalized)
        assert all(np.abs(mean_lab - cnorm['mu']) < [0.1, 0.1, 0.1])
        assert all(np.abs(std_lab - cnorm['sigma']) < [0.1, 0.1, 0.1])

        # Do MASKED normalization to preset standard
        tissue_rgb_normalized = reinhard(
            cfg.tissue_rgb, target_mu=cnorm['mu'], target_sigma=cnorm['sigma'],
            mask_out=cfg.mask_out)

        # check that it matches
        mean_lab, std_lab = lab_mean_std(
            tissue_rgb_normalized, mask_out=cfg.mask_out)
        assert all(np.abs(mean_lab - cnorm['mu']) < [0.1, 0.1, 0.1])
        assert all(np.abs(std_lab - cnorm['sigma']) < [0.1, 0.1, 0.1])

    def test_macenko_normalization(self):
        """Test macenko_pca normalization."""
        stain_unmixing_routine_params = {
            'stains': ['hematoxylin', 'eosin'],
            'stain_unmixing_method': 'macenko_pca',
        }

        # TCGA-A2-A3XS-DX1_xmin21421_ymin37486_.png, Amgad et al, 2019)
        # for macenco (obtained using rgb_separate_stains_macenko_pca()
        # and using reordered such that columns are the order:
        # Hamtoxylin, Eosin, Null
        W_target = np.array([
            [0.5807549, 0.08314027, 0.08213795],
            [0.71681094, 0.90081588, 0.41999816],
            [0.38588316, 0.42616716, -0.90380025]
        ])

        # Macenko - Unmasked, using default, 'idealized' W_target"
        tissue_rgb_normalized = deconvolution_based_normalization(
            cfg.tissue_rgb,
            stain_unmixing_routine_params=stain_unmixing_routine_params)
        assert tuple(
            [int(tissue_rgb_normalized[..., i].mean()) for i in range(3)]
            ) == (183, 121, 212)

        # Macenko - Unmasked, using W_target from good image
        tissue_rgb_normalized = deconvolution_based_normalization(
            cfg.tissue_rgb, W_target=W_target,
            stain_unmixing_routine_params=stain_unmixing_routine_params)
        assert tuple(
            [int(tissue_rgb_normalized[..., i].mean()) for i in range(3)]
            ) == (188, 125, 175)

        # Macenko - Masked, using W_target from good image
        tissue_rgb_normalized = deconvolution_based_normalization(
            cfg.tissue_rgb, W_target=W_target, mask_out=cfg.mask_out,
            stain_unmixing_routine_params=stain_unmixing_routine_params)
        assert tuple(
            [int(tissue_rgb_normalized[..., i].mean()) for i in range(3)]
            ) == (188, 125, 175)


class TestColorAugmentation:
    """Test color augmentation."""

    def test_rgb_perturb_stain_concentration(self):
        """Test rgb_perturb_stain_concentration."""
        # for reproducibility
        np.random.seed(0)

        # Unmasked
        augmented_rgb = rgb_perturb_stain_concentration(cfg.tissue_rgb)
        assert tuple(
            [int(augmented_rgb[..., i].mean()) for i in range(3)]
            ) == (178, 115, 154)

        # Masked
        augmented_rgb = rgb_perturb_stain_concentration(
            cfg.tissue_rgb, mask_out=cfg.mask_out)
        assert tuple(
            [int(augmented_rgb[..., i].mean()) for i in range(3)]
            ) == (174, 101, 139)
