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

from histomicstk.preprocessing.color_deconvolution import stain_color_map

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), '../server')))
from cli_common import utils  # noqa


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
            stain_1_vector=None,
            stain_2='custom',
            stain_2_vector=[0.1, 0.2, 0.3],
        )
        expected = np.array([stain_color_map['hematoxylin'],
                             [0.1, 0.2, 0.3]]).T
        np.testing.assert_allclose(utils.get_stain_matrix(args, 2), expected)
