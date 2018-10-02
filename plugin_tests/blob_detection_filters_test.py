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

import os
import unittest

import numpy as np

from histomicstk.filters.shape import clog, cdog


TEST_DATA_DIR = os.path.join(os.environ['GIRDER_TEST_DATA_PREFIX'],
                             'plugins/HistomicsTK')


class BlobDetectionFiltersTest(unittest.TestCase):

    def test_clog(self):

        im_nuclei_stain = np.load(
            os.path.join(TEST_DATA_DIR, 'Easy1_nuclei_stain.npy'))

        im_nuclei_fgnd_mask = np.load(
            os.path.join(TEST_DATA_DIR, 'Easy1_nuclei_fgnd_mask.npy'))

        sigma_min = 10.0 / np.sqrt(2.0)
        sigma_max = 40.0 / np.sqrt(2.0)

        im_log_max, im_sigma_max = clog(im_nuclei_stain,
                                        im_nuclei_fgnd_mask,
                                        sigma_min, sigma_max)

        im_log_max_gtruth = np.load(
            os.path.join(TEST_DATA_DIR, 'Easy1_clog_max.npy'))

        np.testing.assert_array_almost_equal(
            im_log_max.astype('float16'), im_log_max_gtruth, decimal=4)

        im_sigma_max_gtruth = np.load(
            os.path.join(TEST_DATA_DIR, 'Easy1_clog_sigma_max.npy'))

        np.testing.assert_array_almost_equal(
            im_sigma_max.astype(np.float16), im_sigma_max_gtruth, decimal=4)

    def test_cdog(self):

        im_nuclei_stain = np.load(
            os.path.join(TEST_DATA_DIR, 'Easy1_nuclei_stain.npy'))

        im_nuclei_fgnd_mask = np.load(
            os.path.join(TEST_DATA_DIR, 'Easy1_nuclei_fgnd_mask.npy'))

        sigma_min = 10.0 / np.sqrt(2.0)
        sigma_max = 40.0 / np.sqrt(2.0)

        im_dog_max, im_sigma_max = cdog(im_nuclei_stain,
                                        im_nuclei_fgnd_mask,
                                        sigma_min, sigma_max)

        im_dog_max_gtruth = np.load(
            os.path.join(TEST_DATA_DIR, 'Easy1_cdog_max.npy'))

        np.testing.assert_array_almost_equal(
            im_dog_max.astype(np.float16), im_dog_max_gtruth, decimal=4)

        im_sigma_max_gtruth = np.load(
            os.path.join(TEST_DATA_DIR, 'Easy1_cdog_sigma_max.npy'))

        np.testing.assert_array_almost_equal(
            im_sigma_max.astype(np.float16), im_sigma_max_gtruth, decimal=4)
