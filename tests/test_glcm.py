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

import histomicstk as htk
import numpy as np
import skimage.feature


class TestGLCMMatrixGeneration:

    def test_graycomatrixext(self):

        image = np.array([[0, 0, 1, 1],
                          [0, 0, 1, 1],
                          [0, 2, 2, 2],
                          [2, 2, 3, 3]], dtype=np.uint8)

        # test 0, 45, 90, 135 degree offsets
        res_skim = skimage.feature.greycomatrix(
            image, [1],
            [0, np.pi/4.0, np.pi/2.0, 3*np.pi/4.0],
            levels=4
        )

        res_htk = htk.features.graycomatrixext(
            image, num_levels=4, gray_limits=[0, 3],
            offsets=np.array([[0, 1], [1, 1], [1, 0], [1, -1]])
        )

        np.testing.assert_allclose(np.squeeze(res_htk), np.squeeze(res_skim))

        # test 0, 45, 90, 135 degree offsets - normalized
        res_skim = skimage.feature.greycomatrix(
            image, [1],
            [0, np.pi/4.0, np.pi/2.0, 3*np.pi/4.0],
            levels=4, normed=True
        )

        res_htk = htk.features.graycomatrixext(
            image, num_levels=4, gray_limits=[0, 3], normed=True,
            offsets=np.array([[0, 1], [1, 1], [1, 0], [1, -1]])
        )

        np.testing.assert_allclose(np.squeeze(res_htk), np.squeeze(res_skim))

        # test 0, 45, 90, 135 degree offsets - symmetric
        res_skim = skimage.feature.greycomatrix(
            image, [1],
            [0, np.pi/4.0, np.pi/2.0, 3*np.pi/4.0],
            levels=4, symmetric=True
        )

        res_htk = htk.features.graycomatrixext(
            image, num_levels=4, gray_limits=[0, 3], symmetric=True,
            offsets=np.array([[0, 1], [1, 1], [1, 0], [1, -1]])
        )

        np.testing.assert_allclose(np.squeeze(res_htk), np.squeeze(res_skim))
