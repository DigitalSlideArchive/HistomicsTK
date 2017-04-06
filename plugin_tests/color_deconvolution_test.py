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

from histomicstk.preprocessing import color_deconvolution as htk_dcv


# boiler plate to start and stop the server if needed
def setUpModule():
    base.enabledPlugins.append('HistomicsTK')
    base.startServer()


def tearDownModule():
    base.stopServer()


TEST_DATA_DIR = os.path.join(os.environ['GIRDER_TEST_DATA_PREFIX'],
                             'plugins/HistomicsTK')


class MacenkoTest(base.TestCase):

    def test_macenko(self):
        im_path = os.path.join(TEST_DATA_DIR, 'Easy1.png')
        im = skimage.io.imread(im_path)[..., :3]

        #np.random.seed(1)

        w = htk_dcv.rgb_macenko_stain_matrix(im, 255)

        w_expected = [[ 0.089411,  0.558021, -0.130574],
                      [ 0.837138,  0.729935,  0.546981],
                      [ 0.539635,  0.394725, -0.826899]]

        np.testing.assert_allclose(w, w_expected, atol=1e-6)
