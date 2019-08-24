# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 16:01:26 2019.

@author: tageldim
"""

import os
import numpy as np
from pandas import DataFrame, concat
# import cv2
from shapely.geometry.polygon import Polygon
from shapely.ops import cascaded_union
# from histomicstk.annotations_and_masks.masks_to_annotations_handler import (
from masks_to_annotations_handler import (
    Conditional_Print, _parse_annot_coords)
from annotation_and_mask_utils import (
    parse_slide_annotations_into_table)


# %% =====================================================================


class Polygon_merger_v2(object):
    """Methods to merge contiguous polygons from whole-slide image."""

    def __init__(self, contours_df, **kwargs):
        """Init Polygon_merger object.

        Arguments:
        -----------
        contours_df : pandas DataFrame
            The following columns are needed.

            group : str
                annotation group (ground truth label).
            ymin : int
                minimun y coordinate
            ymax : int
                maximum y coordinate
            xmin : int
                minimum x coordinate
            xmax : int
                maximum x coordinate
            coords_x : str
                vertix x coordinates comma-separated values
            coords_y
                vertix y coordinated comma-separated values
        merge_thresh : int
            how close do the polygons need to be (in pixels) to be merged
        verbose : int
            0 - Do not print to screen
            1 - Print only key messages
            2 - Print everything to screen
        monitorPrefix : str
            text to prepend to printed statements

        """
        # see: https://stackoverflow.com/questions/8187082/how-can-you-set-...
        # class-attributes-from-variable-arguments-kwargs-in-python
        default_attr = {
            'verbose': 1,
            'monitorPrefix': "",
            'merge_thresh': 3,
        }
        more_allowed_attr = ['', ]
        allowed_attr = list(default_attr.keys()) + more_allowed_attr
        default_attr.update(kwargs)
        self.__dict__.update(
            (k, v) for k, v in default_attr.items() if k in allowed_attr)

        # To NOT silently ignore rejected keys
        rejected_keys = set(kwargs.keys()) - set(allowed_attr)
        if rejected_keys:
            raise ValueError(
                "Invalid arguments in constructor:{}".format(rejected_keys))

        # verbosity control
        self.cpr1 = Conditional_Print(verbose=self.verbose == 1)
        self._print1 = self.cpr1._print
        self.cpr2 = Conditional_Print(verbose=self.verbose == 2)
        self._print2 = self.cpr2._print
        self.contkwargs['verbose'] = self.verbose > 1

    # %% =====================================================================

# %% =====================================================================
# %% =====================================================================

# %%===========================================================================
# Constants & prep work
# =============================================================================

import girder_client

APIURL = 'http://candygram.neurology.emory.edu:8080/api/v1/'
SOURCE_SLIDE_ID = '5d5d6910bd4404c6b1f3d893'

gc = girder_client.GirderClient(apiUrl=APIURL)
# gc.authenticate(interactive=True)
gc.authenticate(apiKey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')

# %%===========================================================================

# get and parse slide annotations into dataframe
slide_annotations = gc.get('/annotation/item/5d586d57bd4404c6b1f28640')
contours_df = parse_slide_annotations_into_table(slide_annotations)

# %%===========================================================================

# init polygon merger
pm = Polygon_merger_v2(contours_df, verbose=1)






