#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

from tests import base

import numpy as np
import os
import skimage.io

from histomicstk.preprocessing import color_conversion as htk_cvt
from histomicstk.preprocessing import color_normalization as htk_cn


# boiler plate to start and stop the server if needed
def setUpModule():
    base.enabledPlugins.append('HistomicsTK')
    base.startServer()


def tearDownModule():
    base.stopServer()


TEST_DATA_DIR = os.path.join(os.environ['GIRDER_TEST_DATA_PREFIX'],
                             'plugins/HistomicsTK')


class ReinhardNormalizationTest(base.TestCase):

    def test_normalization(self):

        input_image_file = os.path.join(TEST_DATA_DIR, 'L1.png')

        ref_image_file = os.path.join(TEST_DATA_DIR, 'Easy1.png')

        # read input image
        im_input = skimage.io.imread(input_image_file)[:, :, :3]

        # read reference image
        im_reference = skimage.io.imread(ref_image_file)[:, :, :3]

        # get mean and stddev of reference image in lab space
        mean_ref, std_ref = htk_cvt.lab_mean_std(im_reference)

        # perform color normalization
        im_nmzd = htk_cn.reinhard(im_input, mean_ref, std_ref)

        # transform normalized image to LAB color space
        mean_nmzd, std_nmzd = htk_cvt.lab_mean_std(im_nmzd)

        # check if mean and stddev of normalized and reference images are equal
        np.testing.assert_allclose(mean_nmzd, mean_ref, atol=1e-1)
        np.testing.assert_allclose(std_nmzd, std_ref, atol=1e-1)

    def test_reinhard_sample(self):

        wsi_path = os.path.join(
            TEST_DATA_DIR,
            'TCGA-OR-A5J1-01A-01-TS1.CFE08710-54B8-45B0-86AE-500D6E36D8A5.svs'
        )

        wsi_mean, wsi_stddev = htk_cn.reinhard_sample(
            wsi_path, 20, 0.05, 240)

        gt_mean = [7.54740211, -0.23243189, 0.05317158]
        gt_stddev = [0.96676908, 0.14012439, 0.03045649]

        np.testing.assert_allclose(wsi_mean, gt_mean, atol=1e-2)
        np.testing.assert_allclose(wsi_stddev, gt_stddev, atol=1e-2)
