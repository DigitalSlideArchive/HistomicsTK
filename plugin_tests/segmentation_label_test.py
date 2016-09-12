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

from histomicstk.segmentation import label

# boiler plate to start and stop the server if needed
def setUpModule():
    base.enabledPlugins.append('HistomicsTK')
    base.startServer()


def tearDownModule():
    base.stopServer()


TEST_DATA_DIR = os.path.join(os.environ['GIRDER_TEST_DATA_PREFIX'],
                             'plugins/HistomicsTK')


class TraceBoundsTest(base.TestCase):

    def test_isbf(self):

        mask = np.array([[0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 1, 0, 0],
                         [0, 1, 1, 1, 1, 0],
                         [0, 1, 1, 1, 1, 0],
                         [0, 1, 1, 1, 1, 0],
                         [0, 0, 1, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0]], dtype=np.bool)

        # test isbf with 4 connectivity
        res_X, res_Y = label.TraceBounds(mask, 4)

        cython_X, cython_Y = label.TraceBounds(mask, 4)

        # check if isbf of original X, Y and cython version X, Y are equal
        np.testing.assert_allclose(res_X, cython_X)
        np.testing.assert_allclose(res_Y, cython_Y)
