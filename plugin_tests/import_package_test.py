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

import unittest


# Test import of histomicstk and all of its sub-packages independently
class ImportPackageTest(unittest.TestCase):

    def test_histomicstk(self):
        import histomicstk as htk  # noqa

    def test_features(self):
        from histomicstk import features  # noqa

    def test_filters(self):
        from histomicstk import filters  # noqa

    def test_filters_edge(self):
        from histomicstk.filters import edge  # noqa

    def test_filters_shape(self):
        from histomicstk.filters import shape  # noqa

    def test_preprocessing(self):
        from histomicstk import preprocessing  # noqa

    def test_preprocessing_color_conversion(self):
        from histomicstk.preprocessing import color_conversion  # noqa

    def test_preprocessing_color_deconvolution(self):
        from histomicstk.preprocessing import color_deconvolution  # noqa

    def test_preprocessing_color_normalization(self):
        from histomicstk.preprocessing import color_normalization  # noqa

    def test_segmentation(self):
        from histomicstk import segmentation  # noqa

    def test_segmentation_label(self):
        from histomicstk.segmentation import label  # noqa

    def test_segmentation_level_set(self):
        from histomicstk.segmentation import level_set  # noqa

    def test_segmentation_nuclear(self):
        from histomicstk.segmentation import nuclear  # noqa

    def test_utils(self):
        from histomicstk import utils  # noqa
