import histomicstk.preprocessing.color_deconvolution as htk_cdeconv
import histomicstk.filters.shape as htk_shape_filters

import numpy as np
import scipy as sp

import skimage.color
import skimage.filters
import skimage.io
import skimage.measure
import skimage.morphology



def membrane_detection(I):
    """Performs membrane detection.

    Parameters
    ----------
    I : array_like
        A hematoxylin intensity image obtained from ColorDeconvolution.

    Returns
    -------

    """

    im_input = skimage.io.imread(I)[:, :, :3]
