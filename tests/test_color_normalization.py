#!/usr/bin/env python

###############################################################################
#  Copyright Kitware Inc.
#
#  Licensed under the Apache License, Version 2.0 ( the "License" );
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

import collections
import numpy as np
import os
from histomicstk.preprocessing import color_normalization as htk_cn
from histomicstk.cli import utils as cli_utils

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

        gt_mean = [8.896134, -0.074579, 0.022006]
        gt_stddev = [0.612143, 0.122667, 0.021361]

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
