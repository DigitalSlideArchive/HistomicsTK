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

import numpy as np

from tests import base

from histomicstk.features import compute_global_graph_features as cggf
from histomicstk.features.compute_global_graph_features import (
    Props, PolyProps, TriProps, DensityProps, PopStats
)


# boiler plate to start and stop the server
def setUpModule():
    base.enabledPlugins.append('HistomicsTK')
    base.startServer()


def tearDownModule():
    base.stopServer()


class GlobalGraphFeaturesTest(base.TestCase):
    def _assert_equal_nested(self, actual, expected, rtol=1e-6):
        def recur(actual, expected):
            if isinstance(expected, int):
                self.assertEqual(actual, expected)
            elif isinstance(expected, float):
                assert (actual == expected
                        or np.isnan(actual) and np.isnan(actual)
                        or abs(actual / expected - 1) <= rtol), \
                        "Expected {} but got {}!".format(expected, actual)
            elif isinstance(expected, tuple):
                self.assertEqual(len(actual), len(expected))
                for a, e in zip(actual, expected):
                    recur(a, e)
            elif isinstance(expected, dict):
                self.assertEqual(len(actual), len(expected))
                actual = sorted(actual.items())
                expected = sorted(expected.items())
                for (k1, v1), (k2, v2) in zip(actual, expected):
                    recur(k1, k2)
                    recur(v1, v2)
            else:
                raise ValueError('Unknown type {}!'.format(type(expected)))
        recur(actual, expected)

    def testSimple(self):
        data = np.array([[-1,-1],[-1,1],[1,-1],[1,1],[-.5,-.5],[.5,.5]])
        actual = cggf(data, scale=0.7, neighbor_counts=(3, 5))
        expected = Props(
            voronoi=PolyProps(
                area=PopStats(
                    mean=1.6875,
                    stddev=0.0,
                    minmaxr=1.0,
                    disorder=0.0,
                ),
                peri=PopStats(
                    mean=5.5536887604657483,
                    stddev=0.0,
                    minmaxr=1.0,
                    disorder=0.0,
                ),
                max_dist=PopStats(
                    mean=2.1213203435596424,
                    stddev=0.0,
                    minmaxr=1.0,
                    disorder=0.0,
                ),
            ),
            delaunay=TriProps(
                sides=PopStats(
                    mean=1.5593620404620863,
                    stddev=0.45249714157059079,
                    minmaxr=0.35355339059327379,
                    disorder=0.22491491731216068,
                ),
                area=PopStats(
                    mean=0.66666666666666663,
                    stddev=0.23570226039551584,
                    minmaxr=0.5,
                    disorder=0.26120387496374148,
                ),
            ),
            mst_branches=PopStats(
                mean=1.198140956982914,
                stddev=0.40553452035562559,
                minmaxr=0.44721359549995793,
                disorder=0.2528781702322036,
            ),
            density=DensityProps(
                neighbors_in_distance={
                    0.7: PopStats(
                        mean=0.0,
                        stddev=0.0,
                        minmaxr=np.nan,
                        disorder=np.nan,
                    ),
                    1.4: PopStats(
                        mean=0.66666666666666663,
                        stddev=0.47140452079103168,
                        minmaxr=0.0,
                        disorder=0.41421356237309503,
                    ),
                    2.1: PopStats(
                        mean=3.6666666666666665,
                        stddev=0.47140452079103168,
                        minmaxr=0.75,
                        disorder=0.11391890072356341,
                    ),
                    2.8: PopStats(
                        mean=4.333333333333333,
                        stddev=0.47140452079103168,
                        minmaxr=0.80000000000000004,
                        disorder=0.098112432999103216,
                    ),
                    3.5: PopStats(
                        mean=5.0,
                        stddev=0.0,
                        minmaxr=1.0,
                        disorder=0.0,
                    ),
                },
                distance_for_neighbors={
                    3: PopStats(
                        mean=1.860379610028063,
                        stddev=0.19745304908213343,
                        minmaxr=0.79056941504209488,
                        disorder=0.095951946436456992,
                    ),
                    5: PopStats(
                        mean=2.5927248643506742,
                        stddev=0.33333333333333354,
                        minmaxr=0.74999999999999989,
                        disorder=0.11391890072356348,
                    ),
                },
            ),
        )
        self._assert_equal_nested(actual, expected)
