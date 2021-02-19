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

import numpy as np
from histomicstk.preprocessing.color_conversion import rgb_to_od
from histomicstk.preprocessing.color_conversion import od_to_rgb
from histomicstk.preprocessing.color_conversion import rgb_to_lab
from histomicstk.preprocessing.color_conversion import lab_to_rgb


class TestColorConversion:

    def test_rgb_to_od(self):

        np.testing.assert_array_almost_equal(
            np.round(rgb_to_od(np.zeros((3, 3, 3)) + 117.0), 4),
            np.zeros((3, 3, 3)) + 35.6158
        )

        # check corner cases
        np.testing.assert_array_almost_equal(
            rgb_to_od(np.zeros((3, 3, 3)) + 255.0),
            np.zeros((3, 3, 3))
        )

        np.testing.assert_array_almost_equal(
            rgb_to_od(np.zeros((3, 3, 3))),
            np.zeros((3, 3, 3)) + 255.0
        )

    def test_od_to_rgb(self):

        np.testing.assert_array_almost_equal(
            od_to_rgb(np.zeros((3, 3, 3)) + 35.6158),
            np.zeros((3, 3, 3)) + 116.99987889
        )

        # check corner cases
        np.testing.assert_array_almost_equal(
            od_to_rgb(np.zeros((3, 3, 3))),
            np.zeros((3, 3, 3)) + 255.0
        )

        np.testing.assert_array_almost_equal(
            od_to_rgb(np.zeros((3, 3, 3)) + 255.0),
            np.zeros((3, 3, 3))
        )

    def test_rgb_to_od_to_rgb(self):

        np.random.seed(1)

        im_rand = np.random.randint(0, 255, (10, 10, 3))

        np.testing.assert_array_almost_equal(
            od_to_rgb(rgb_to_od(im_rand)),
            im_rand
        )

    def test_rgb_to_lab_to_rgb(self):

        np.random.seed(1)

        im_rand = np.random.randint(0, 255, (10, 10, 3))

        np.testing.assert_array_almost_equal(
            np.round(lab_to_rgb(rgb_to_lab(im_rand))),
            im_rand
        )
