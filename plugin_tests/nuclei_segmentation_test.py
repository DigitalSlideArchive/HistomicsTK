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

import os
import sys

import scipy as sp
import numpy as np
import skimage.io

import histomicstk.preprocessing.color_conversion as htk_cvt
import histomicstk.preprocessing.color_normalization as htk_cnorm
import histomicstk.preprocessing.color_deconvolution as htk_cdeconv
import histomicstk.filters.shape as htk_shape_filters
import histomicstk.segmentation as htk_seg


# boiler plate to start and stop the server if needed
def setUpModule():
    base.enabledPlugins.append('HistomicsTK')
    base.startServer()


def tearDownModule():
    base.stopServer()


TEST_DATA_DIR = os.path.join(os.environ['GIRDER_TEST_DATA_PREFIX'],
                             'plugins/HistomicsTK')


class NucleiSegmentationTest(base.TestCase):

    def test_segment_nuclei_kofahi(self):

        input_image_file = os.path.join(TEST_DATA_DIR, 'Easy1.png')

        ref_image_file = os.path.join(TEST_DATA_DIR, 'L1.png')

        # read input image
        im_input = skimage.io.imread(input_image_file)[:, :, :3]

        # read reference image
        im_reference = skimage.io.imread(ref_image_file)[:, :, :3]

        # get mean and stddev of reference image in lab space
        mean_ref, std_ref = htk_cvt.lab_mean_std(im_reference)

        # perform color normalization
        im_nmzd = htk_cnorm.reinhard(im_input, mean_ref, std_ref)

        # perform color decovolution
        stain_color_map = {
            'hematoxylin': [0.65, 0.70, 0.29],
            'eosin': [0.07, 0.99, 0.11],
            'dab': [0.27, 0.57, 0.78],
            'null': [0.0, 0.0, 0.0]
        }

        w = np.array([stain_color_map['hematoxylin'],
                      stain_color_map['eosin'],
                      stain_color_map['null']]).T

        im_stains = htk_cdeconv.color_deconvolution(im_nmzd, w).Stains

        im_nuclei_stain = im_stains[:, :, 0].astype(np.float)

        # segment foreground (assumes nuclei are darker on a bright background)
        im_nuclei_fgnd_mask = sp.ndimage.morphology.binary_fill_holes(
            im_nuclei_stain < 160)

        # run adaptive multi-scale LoG filter
        im_log = htk_shape_filters.clog(im_nuclei_stain, im_nuclei_fgnd_mask,
                                        sigma_min=20 / np.sqrt(2),
                                        sigma_max=30 / np.sqrt(2))

        # apply local maximum clustering
        im_nuclei_seg_mask, seeds, max = htk_seg.nuclear.max_clustering(
            im_log, im_nuclei_fgnd_mask, 10)

        # filter out small objects
        im_nuclei_seg_mask = htk_seg.label.area_open(
            im_nuclei_seg_mask, 80).astype(np.int)

        # check if segmentation mask matches ground truth
        gtruth_mask_file = os.path.join(TEST_DATA_DIR,
                                        'Easy1_nuclei_seg_kofahi.npy')

        im_gtruth_mask = np.load(gtruth_mask_file)

        sys.stderr.write('%r\n' % im_nuclei_seg_mask)
        sys.stderr.write('%r\n' % im_gtruth_mask)

        np.testing.assert_allclose(im_nuclei_seg_mask, im_gtruth_mask)
