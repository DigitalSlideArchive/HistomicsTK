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

import os
import numpy as np
from histomicstk.filters.shape import clog, cdog
from skimage.feature import peak_local_max

from .datastore import datastore


def assert_array_almost_equal_neighborhood_lines(im, gt, decimal=4):
    """Wrapper around assert_array_almost_equal to work around scikit-image bug.

    `skimage.transform.resize()` can return different results on different platforms. This
    is an uncorrected bug mostly due to some floating point computation differences on
    different platforms, hardware, or numerical computation library backend [1].

    Due to that bug, the image (array) returned by the `resize()` function may have a few
    lines in the result that differ from the ground truth computed on a different computer.

    The work around implemented in this function compares each line of the input image with
    each line of a ground truth image, but also compares the line before and the line after.
    Only the minimum error is kept. This allows to ignore 'rogue' lines in the image, that
    are different only due to that scikit-image bug.

    [1] https://github.com/scikit-image/scikit-image/issues/3445
    """
    shift = [-1, 0, 1]
    min_array = np.full_like(im, np.inf)
    for s in shift:
        rolled_image = np.roll(im, s, 1)
        for s in shift:
            rolled_gt = np.roll(gt, s, 1)
            abs_array = np.abs(rolled_gt - rolled_image)
            min_array = np.minimum(min_array, abs_array)
    np.testing.assert_array_almost_equal(min_array, np.zeros_like(im), decimal=decimal)


def _sort_list(input_list):
    return input_list[input_list[:, 0].argsort()]


def compare_maxima(input_im, gtruth_im, min_distance=10, threshold_abs=20):
    """Compares image maxima

    Compare that the maxima found in an image matches the maxima found in a ground truth image.
    This function is a wrapper around `skimage.feature.peak_local_max()`. It calls this function
    on both images that are passed as arguments, and asserts if the resulting maxima arrays
    returned by this function match.
    """
    gtruth_coordinates = _sort_list(peak_local_max(gtruth_im, min_distance=min_distance,
                                                   threshold_abs=threshold_abs))
    input_coordinates = _sort_list(peak_local_max(input_im, min_distance=min_distance,
                                                  threshold_abs=threshold_abs))
    np.testing.assert_array_equal(gtruth_coordinates, input_coordinates)


class TestBlobDetectionFilters:

    def test_clog(self):

        im_nuclei_stain_data = np.load(
            datastore.fetch('Easy1_nuclei_stain.npz'))
        im_nuclei_stain = im_nuclei_stain_data['Easy1_nuclei_stain']

        im_nuclei_fgnd_mask_data = np.load(
            datastore.fetch('Easy1_nuclei_fgnd_mask.npz'))
        im_nuclei_fgnd_mask = im_nuclei_fgnd_mask_data['Easy1_nuclei_fgnd_mask']

        sigma_min = 10.0 / np.sqrt(2.0)
        sigma_max = 40.0 / np.sqrt(2.0)

        im_log_max, im_sigma_max = clog(
            im_input=im_nuclei_stain, im_mask=im_nuclei_fgnd_mask,
            sigma_min=sigma_min, sigma_max=sigma_max)

        im_log_max_gtruth_data = np.load(os.path.join(datastore.fetch(
            'Easy1_clog_max.npz')))
        im_log_max_gtruth = im_log_max_gtruth_data['Easy1_clog_max']
        assert_array_almost_equal_neighborhood_lines(im_log_max, im_log_max_gtruth, decimal=4)
        compare_maxima(im_log_max, im_log_max_gtruth)

        im_sigma_max_gtruth_data = np.load(os.path.join(datastore.fetch(
            'Easy1_clog_sigma_max.npz')))
        im_sigma_max_gtruth = im_sigma_max_gtruth_data['Easy1_clog_sigma_max']
        assert_array_almost_equal_neighborhood_lines(im_sigma_max, im_sigma_max_gtruth, decimal=4)

    def test_cdog(self):

        im_nuclei_stain_data = np.load(
            os.path.join(datastore.fetch('Easy1_nuclei_stain.npz')))
        im_nuclei_stain = im_nuclei_stain_data['Easy1_nuclei_stain']

        im_nuclei_fgnd_mask_data = np.load(
            os.path.join(datastore.fetch('Easy1_nuclei_fgnd_mask.npz')))
        im_nuclei_fgnd_mask = im_nuclei_fgnd_mask_data['Easy1_nuclei_fgnd_mask']

        sigma_min = 10.0 / np.sqrt(2.0)
        sigma_max = 40.0 / np.sqrt(2.0)

        im_dog_max, im_sigma_max = cdog(im_nuclei_stain,
                                        im_nuclei_fgnd_mask,
                                        sigma_min, sigma_max)

        im_dog_max_gtruth_data = np.load(
            os.path.join(datastore.fetch('Easy1_cdog_max.npz')))
        im_dog_max_gtruth = im_dog_max_gtruth_data['Easy1_cdog_max']
        assert_array_almost_equal_neighborhood_lines(im_dog_max, im_dog_max_gtruth, decimal=4)
        compare_maxima(im_dog_max, im_dog_max_gtruth)

        im_sigma_max_gtruth_data = np.load(
            os.path.join(datastore.fetch('Easy1_cdog_sigma_max.npz')))
        im_sigma_max_gtruth = im_sigma_max_gtruth_data['Easy1_cdog_sigma_max']
        assert_array_almost_equal_neighborhood_lines(im_sigma_max, im_sigma_max_gtruth, decimal=4)
