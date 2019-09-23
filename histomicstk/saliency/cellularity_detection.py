#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 21:17:43 2019

@author: mtageld
"""

from pandas import DataFrame, concat
from skimage.color import rgb2gray
from skimage.segmentation import slic
from skimage.transform import resize
from sklearn.mixture import GaussianMixture
from skimage.measure import regionprops
from matplotlib import cm

from histomicstk.saliency.tissue_detection import (
    get_slide_thumbnail, get_tissue_mask, _deconv_color)
from histomicstk.annotations_and_masks.masks_to_annotations_handler import (
    get_contours_from_mask, get_annotation_documents_from_contours)
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    get_image_from_htk_response)
from histomicstk.features.compute_intensity_features import (
    compute_intensity_features)
from histomicstk.features.compute_haralick_features import (
    compute_haralick_features)