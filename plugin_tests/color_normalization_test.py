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

import histomicstk as htk
import numpy as np
import skimage.io


# boiler plate to start and stop the server
def setUpModule():
    base.enabledPlugins.append('HistomicsTK')
    base.startServer()


def tearDownModule():
    base.stopServer()


class HistomicsTKColorNormalizationTest(base.TestCase):

    def testReinhardNorm(self):

        # will be changed once EXTERNAL_DATA works for data.kitware.com
        inputImageFile = ('https://data.kitware.com/api/v1/file/'
                          '576ad3a68d777f1ecd6702f8/download')

        refImageFile = ('https://data.kitware.com/api/v1/file/'
                        '576ad39b8d777f1ecd6702f2/download')

        # read input image
        imInput = skimage.io.imread(inputImageFile)[:, :, :3]

        # read reference image
        imReference = skimage.io.imread(refImageFile)[:, :, :3]

        # transform reference image to LAB color space
        imReferenceLAB = htk.RudermanLABFwd(imReference)

        # compute mean and stddev of reference image in LAB color space
        meanRef = np.zeros(3)
        stdRef = np.zeros(3)

        for i in range(3):
            meanRef[i] = imReferenceLAB[:, :, i].mean()
            stdRef[i] = (imReferenceLAB[:, :, i] - meanRef[i]).std()

        # perform color normalization
        imNmzd = htk.ReinhardNorm(imInput, meanRef, stdRef)

        # transform reference image to LAB color space
        imNmzdLAB = htk.RudermanLABFwd(imNmzd)

        # compute mean and stddev of normalized input in LAB color space
        meanNmzd = np.zeros(3)
        stdNmzd = np.zeros(3)

        for i in range(3):
            meanNmzd[i] = imNmzdLAB[:, :, i].mean()
            stdNmzd[i] = (imNmzdLAB[:, :, i] - meanNmzd[i]).std()

        # check if mean and stddev of normalized and reference images are equal
        self.assertTrue(np.allclose(meanNmzd, meanRef))
        self.assertTrue(np.allclose(stdNmzd, stdRef))
