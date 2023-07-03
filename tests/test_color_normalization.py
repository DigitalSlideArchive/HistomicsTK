import collections
import os

import numpy as np

from histomicstk.cli import utils as cli_utils
from histomicstk.preprocessing import color_normalization as htk_cn

from .datastore import datastore


class TestReinhardNormalization:

    def test_reinhard_stats(self):

        wsi_path = os.path.join(datastore.fetch(
            'sample_svs_image.TCGA-DU-6399-01A-01-TS1.e8eb65de-d63e-42db-af6f-14fefbbdf7bd.svs'  # noqa
        ))

        np.random.seed(1)

        # create dask client
        args = {
            # In Python 3 unittesting, the scheduler fails if it uses processes
            'scheduler': 'multithreading',  # None,
            'num_workers': -1,
            'num_threads_per_worker': 1,
        }

        args = collections.namedtuple('Parameters', args.keys())(**args)

        cli_utils.create_dask_client(args)

        # compute reinhard stats
        wsi_mean, wsi_stddev = htk_cn.reinhard_stats(
            wsi_path, 0.1, magnification=20)

        # Without ICC correction
        # gt_mean = [8.896134, -0.074579, 0.022006]
        # gt_stddev = [0.612143, 0.122667, 0.021361]

        # With icc correction
        gt_mean = [8.977143, -0.082973, 0.021126]
        gt_stddev = [0.561896, 0.124711, 0.025968]

        np.testing.assert_allclose(wsi_mean, gt_mean, atol=1e-2)
        np.testing.assert_allclose(wsi_stddev, gt_stddev, atol=1e-2)


class TestBackgroundIntensity:

    def test_background_intensity(self):
        wsi_path = os.path.join(datastore.fetch(
            'sample_svs_image.TCGA-DU-6399-01A-01-TS1.e8eb65de-d63e-42db-af6f-14fefbbdf7bd.svs'
        ))

        np.random.seed(1)

        # create dask client
        args = {
            # In Python 3 unittesting, the scheduler fails if it uses processes
            'scheduler': 'multithreading',  # None,
            'num_workers': -1,
            'num_threads_per_worker': 1,
        }

        args = collections.namedtuple('Parameters', args.keys())(**args)

        cli_utils.create_dask_client(args)

        # compute background intensity
        I_0 = htk_cn.background_intensity(wsi_path,
                                          sample_approximate_total=5000)

        np.testing.assert_allclose(I_0, [242, 244, 241], atol=1)


class TestReinhardNormalizationInvert:

    def test_reinhard_stats(self):

        wsi_path = os.path.join(datastore.fetch(
            'sample_svs_image.TCGA-DU-6399-01A-01-TS1.e8eb65de-d63e-42db-af6f-14fefbbdf7bd.svs'  # noqa
        ))

        np.random.seed(1)

        # create dask client
        args = {
            # In Python 3 unittesting, the scheduler fails if it uses processes
            'scheduler': 'multithreading',  # None,
            'num_workers': -1,
            'num_threads_per_worker': 1,
        }

        args = collections.namedtuple('Parameters', args.keys())(**args)

        cli_utils.create_dask_client(args)

        # compute reinhard stats
        wsi_mean, wsi_stddev = htk_cn.reinhard_stats(
            wsi_path, 0.1, magnification=20, invert_image=True)

        # With icc correction
        gt_mean = [3.95768563, -0.15254668, 0.02557574]
        gt_stddev = [0.4689624, 0.12877607, 0.03364667]

        np.testing.assert_allclose(wsi_mean, gt_mean, atol=1e-2)
        np.testing.assert_allclose(wsi_stddev, gt_stddev, atol=1e-2)
