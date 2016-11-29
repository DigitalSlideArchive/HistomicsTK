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

from histomicstk.segmentation.label import trace_boundaries


# boiler plate to start and stop the server if needed
def setUpModule():
    base.enabledPlugins.append('HistomicsTK')
    base.startServer()


def tearDownModule():
    base.stopServer()


TEST_DATA_DIR = os.path.join(os.environ['GIRDER_TEST_DATA_PREFIX'],
                             'plugins/HistomicsTK')


class TraceBoundaryTest(base.TestCase):

    def test_trace_boundary(self):

        # test moore neighbor algorithm

        m_neighbor = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                               [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                               [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                               [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                               [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                               [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.bool)

        m_neighbor = np.ascontiguousarray(m_neighbor, dtype=np.int)

        # refenece neighbors for isbf
        rx_isbf = [1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 8, 9, 9, 8, 7, 7, 7,
                   7, 6, 6, 5, 5, 5, 4, 4, 3, 3, 3, 3, 2, 1]
        ry_isbf = [7, 8, 8, 7, 6, 6, 6, 6, 6, 7, 8, 8, 7, 7, 6, 5, 4,
                   3, 3, 2, 2, 1, 2, 2, 3, 3, 4, 5, 6, 7, 7]

        # refenece neighbors for moore
        rx_moore = [1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 8, 9, 9, 8, 7, 7, 7,
                    7, 6, 5, 4, 3, 3, 3, 3, 2, 1]
        ry_moore = [7, 8, 8, 7, 6, 6, 6, 6, 6, 7, 8, 8, 7, 7, 6, 5, 4,
                    3, 2, 1, 2, 3, 4, 5, 6, 7, 7]

        output_isbf = trace_boundaries(m_neighbor)

        np.testing.assert_allclose(rx_isbf, output_isbf[0][1])
        np.testing.assert_allclose(ry_isbf, output_isbf[0][0])

        output_moore = trace_boundaries(m_neighbor, 8)

        np.testing.assert_allclose(rx_moore, output_moore[0][1])
        np.testing.assert_allclose(ry_moore, output_moore[0][0])
