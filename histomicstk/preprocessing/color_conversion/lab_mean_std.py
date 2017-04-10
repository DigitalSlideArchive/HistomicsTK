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

from .rgb_to_lab import rgb_to_lab


def lab_mean_std(im_input):
    """Computes the mean and standard deviation of the intensities of each
    channel of the given RGB image in LAB color space. The outputs of this
    function are needed for reinhard color normalization.

    Parameters
    ----------
    im_input : array_like
        An RGB image

    Returns
    -------
    mean_lab : array_like
        A 3-element array containing the mean of each channel of the input RGB
        in LAB color space.

    std_lab : array_like
        A 3-element array containing the standard deviation of each channel
        of the input RGB in LAB color space.

    See Also
    --------
    histomicstk.preprocessing.color_conversion.rgb_to_lab,
    histomicstk.preprocessing.color_conversion.reinhard

    References
    ----------
    .. [#] E. Reinhard, M. Adhikhmin, B. Gooch, P. Shirley, "Color transfer
       between images," in IEEE Computer Graphics and Applications, vol.21,
       no.5,pp.34-41, 2001.
    .. [#] D. Ruderman, T. Cronin, and C. Chiao, "Statistics of cone
       responses to natural images: implications for visual coding,"
       J. Opt. Soc. Am. A vol.15, pp.2036-2045, 1998.

    """
    im_lab = rgb_to_lab(im_input)

    mean_lab = np.zeros(3)
    std_lab = np.zeros(3)

    for i in range(3):
        mean_lab[i] = im_lab[:, :, i].mean()
        std_lab[i] = (im_lab[:, :, i] - mean_lab[i]).std()

    return mean_lab, std_lab
