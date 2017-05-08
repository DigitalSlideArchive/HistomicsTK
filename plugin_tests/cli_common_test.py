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

# This is to serve as an example for how to create a server-side test in a
# girder plugin, it is not meant to be useful.

from tests import base

import os
import sys

import numpy as np
import skimage.io

import large_image

from histomicstk.preprocessing.color_deconvolution import stain_color_map

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), '../server')))
from cli_common import cli_utils  # noqa

TEST_DATA_DIR = os.path.join(os.environ['GIRDER_TEST_DATA_PREFIX'], 'plugins/HistomicsTK')


# boiler plate to start and stop the server
def setUpModule():
    base.enabledPlugins.append('HistomicsTK')
    base.startServer()


def tearDownModule():
    base.stopServer()


class Namespace(object):
    pass


class CliCommonTest(base.TestCase):

    def test_get_stain_matrix(self):

        args = Namespace()
        args.__dict__ = dict(
            stain_1='hematoxylin',
            stain_1_vector=[-1, -1, -1],
            stain_2='custom',
            stain_2_vector=[0.1, 0.2, 0.3],
        )
        expected = np.array([stain_color_map['hematoxylin'],
                             [0.1, 0.2, 0.3]]).T
        np.testing.assert_allclose(cli_utils.get_stain_matrix(args, 2),
                                   expected)

    def test_get_region_dict(self):

        ts = large_image.getTileSource(os.path.join(TEST_DATA_DIR,
                                                    'Easy1.png'))

        result = cli_utils.get_region_dict([-1, -1, -1, -1], 2000, ts)
        expected = {}
        assert result == expected, "Expected {}, got {}".format(expected,
                                                                result)

        result = cli_utils.get_region_dict([100, 110, 250, 240], 500, ts)
        expected = dict(region=dict(left=100, top=110, width=250, height=240))
        assert result == expected, "Expected {}, got {}".format(expected,
                                                                result)

    def test_segment_wsi_foreground_at_low_res(self):

        wsi_path = os.path.join(
            TEST_DATA_DIR,
            'TCGA-02-0010-01Z-00-DX4.07de2e55-a8fe-40ee-9e98-bcb78050b9f7.svs'
        )

        ts = large_image.getTileSource(wsi_path)

        im_fgnd_mask_lres, fgnd_seg_scale = \
            cli_utils.segment_wsi_foreground_at_low_res(ts)

        np.testing.assert_equal(fgnd_seg_scale['magnification'], 1.25)

        fgnd_mask_gtruth_file = os.path.join(
            TEST_DATA_DIR,
            'TCGA-02-0010-01Z-00-DX4.07de2e55-a8fe-40ee-9e98-bcb78050b9f7_fgnd_mask_lres.png'  # noqa
        )

        im_fgnd_mask_lres_gtruth = skimage.io.imread(
            fgnd_mask_gtruth_file) / 255

        np.testing.assert_array_equal(im_fgnd_mask_lres,
                                      im_fgnd_mask_lres_gtruth)
