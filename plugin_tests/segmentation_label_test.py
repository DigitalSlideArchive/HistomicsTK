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

from histomicstk.segmentation.label import trace_boundary


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

        # test trace boundary using isbf algorithm

        # refenece left neighbor
        rx = [2, 1, 2]
        ry = [1, 1, 1]

        # test left neighbor
        m_left_neighbor = np.array([[0, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 0, 0]], dtype=np.bool)
        x = 2
        y = 1

        by, bx = trace_boundary(m_left_neighbor, Connectivity=4, XStart=y,
                                YStart=x)

        np.testing.assert_allclose(rx, bx)
        np.testing.assert_allclose(ry, by)

        # refenece inner-outer corner at the left-rear
        rx = [2, 1, 2]
        ry = [2, 1, 2]

        # test inner-outer corner at the left-rear
        m_inner_outer_corner_left_rear = np.array([[0, 0, 0, 0],
                                                   [0, 1, 0, 0],
                                                   [0, 0, 1, 0],
                                                   [0, 0, 0, 0]],
                                                  dtype=np.bool)
        x = 2
        y = 2

        by, bx = trace_boundary(m_inner_outer_corner_left_rear,
                                Connectivity=4, XStart=y, YStart=x)

        np.testing.assert_allclose(rx, bx)
        np.testing.assert_allclose(ry, by)

        # refenece inner-outer corner at the front-left
        rx = [2, 1, 2]
        ry = [1, 2, 1]

        # test inner-outer corner at the front-left
        m_inner_outer_corner_front_left = np.array([[0, 0, 0, 0],
                                                    [0, 0, 1, 0],
                                                    [0, 1, 0, 0],
                                                    [0, 0, 0, 0]],
                                                   dtype=np.bool)
        x = 2
        y = 1

        by, bx = trace_boundary(m_inner_outer_corner_front_left,
                                Connectivity=4, XStart=y, YStart=x)

        np.testing.assert_allclose(rx, bx)
        np.testing.assert_allclose(ry, by)

        # refenece inner corner at the front
        rx = [2, 2, 1, 2, 2]
        ry = [1, 2, 2, 2, 1]

        # test inner corner at the front
        m_inner_corner_front = np.array([[0, 0, 0, 0],
                                         [0, 0, 1, 0],
                                         [0, 1, 1, 0],
                                         [0, 0, 0, 0]], dtype=np.bool)
        x = 2
        y = 1

        by, bx = trace_boundary(m_inner_corner_front, Connectivity=4,
                                XStart=y, YStart=x)

        np.testing.assert_allclose(rx, bx)
        np.testing.assert_allclose(ry, by)

        # refenece front neighbor
        rx = [1, 1, 1]
        ry = [1, 2, 1]

        # test front neighbor
        m_front_neighbor = np.array([[0, 0, 0, 0],
                                     [0, 1, 1, 0],
                                     [0, 0, 0, 0],
                                     [0, 0, 0, 0]], dtype=np.bool)

        x = 1
        y = 1

        by, bx = trace_boundary(m_front_neighbor, Connectivity=4,
                                XStart=y, YStart=x)

        np.testing.assert_allclose(rx, bx)
        np.testing.assert_allclose(ry, by)

        # refenece front neighbor
        rx = [1, 2, 1]
        ry = [1, 2, 1]

        # test outer corner
        m_outer_corner = np.array([[0, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 0]], dtype=np.bool)
        x = 1
        y = 1

        by, bx = trace_boundary(m_outer_corner, Connectivity=4,
                                XStart=y, YStart=x)

        np.testing.assert_allclose(rx, bx)
        np.testing.assert_allclose(ry, by)

        # test trace boundary using moore neighbor algorithm

        # refenece neighbors
        rx = [2, 1, 2]
        ry = [2, 1, 2]

        # test NW
        m_neighbor = np.array([[0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0],
                               [0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0]], dtype=np.bool)
        x = 2
        y = 2

        by, bx = trace_boundary(m_neighbor, Connectivity=8,
                                XStart=y, YStart=x)

        np.testing.assert_allclose(rx, bx)
        np.testing.assert_allclose(ry, by)

        # refenece neighbors
        rx = [2, 1, 2]
        ry = [2, 2, 2]

        # test N
        m_neighbor = np.array([[0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0],
                               [0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0]], dtype=np.bool)
        x = 2
        y = 2

        by, bx = trace_boundary(m_neighbor, Connectivity=8,
                                XStart=y, YStart=x)

        np.testing.assert_allclose(rx, bx)
        np.testing.assert_allclose(ry, by)

        # refenece neighbors
        rx = [2, 1, 2]
        ry = [2, 3, 2]

        # test NE
        m_neighbor = np.array([[0, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0],
                               [0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0]], dtype=np.bool)
        x = 2
        y = 2

        by, bx = trace_boundary(m_neighbor, Connectivity=8,
                                XStart=y, YStart=x)

        np.testing.assert_allclose(rx, bx)
        np.testing.assert_allclose(ry, by)

        # refenece neighbors
        rx = [2, 2, 2]
        ry = [2, 3, 2]

        # test E
        m_neighbor = np.array([[0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 1, 1, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0]], dtype=np.bool)
        x = 2
        y = 2

        by, bx = trace_boundary(m_neighbor, Connectivity=8,
                                XStart=y, YStart=x)

        np.testing.assert_allclose(rx, bx)
        np.testing.assert_allclose(ry, by)

        # refenece neighbors
        rx = [2, 3, 2]
        ry = [2, 3, 2]

        # test SE
        m_neighbor = np.array([[0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0],
                               [0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0]], dtype=np.bool)
        x = 2
        y = 2

        by, bx = trace_boundary(m_neighbor, Connectivity=8,
                                XStart=y, YStart=x)

        np.testing.assert_allclose(rx, bx)
        np.testing.assert_allclose(ry, by)

        # refenece neighbors
        rx = [2, 3, 2]
        ry = [2, 2, 2]

        # test S
        m_neighbor = np.array([[0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0],
                               [0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0]], dtype=np.bool)
        x = 2
        y = 2

        by, bx = trace_boundary(m_neighbor, Connectivity=8,
                                XStart=y, YStart=x)

        np.testing.assert_allclose(rx, bx)
        np.testing.assert_allclose(ry, by)

        # refenece neighbors
        rx = [2, 3, 2]
        ry = [2, 1, 2]

        # test SW
        m_neighbor = np.array([[0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0],
                               [0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0]], dtype=np.bool)
        x = 2
        y = 2

        by, bx = trace_boundary(m_neighbor, Connectivity=8,
                                XStart=y, YStart=x)

        np.testing.assert_allclose(rx, bx)
        np.testing.assert_allclose(ry, by)

        # refenece neighbors
        rx = [2, 2, 2]
        ry = [2, 1, 2]

        # test W
        m_neighbor = np.array([[0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 1, 1, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0]], dtype=np.bool)
        x = 2
        y = 2

        by, bx = trace_boundary(m_neighbor, Connectivity=8,
                                XStart=y, YStart=x)

        np.testing.assert_allclose(rx, bx)
        np.testing.assert_allclose(ry, by)
