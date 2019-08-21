# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 16:25:06 2019

@author: tageldim
"""

import numpy as np
from pandas import DataFrame, concat
import cv2
from shapely.geometry.polygon import Polygon

# %% =====================================================================


class Conditional_Print(object):
    """Print to screen if certain conditions are satisfied (Internal)."""

    def __init__(self, verbose=True):
        """Init. This is for PEP compliance."""
        self.verbose = verbose

    def _print(self, text):
        if self.verbose:
            print(text)

# %% =====================================================================


# %%===========================================================================
# Constants & prep work
# =============================================================================

import os
import girder_client
from pandas import read_csv

APIURL = 'http://candygram.neurology.emory.edu:8080/api/v1/'
SAMPLE_SLIDE_ID = '5d586d76bd4404c6b1f286ae'

gc = girder_client.GirderClient(apiUrl=APIURL)
# gc.authenticate(interactive=True)
gc.authenticate(apiKey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')

# read GTCodes dataframe
PTESTS_PATH = os.path.join(os.getcwd(), '..', '..', 'plugin_tests')
GTCODE_PATH = os.path.join(PTESTS_PATH, 'test_files', 'sample_GTcodes.csv')
GTCodes_df = read_csv(GTCODE_PATH)
GTCodes_df.index = GTCodes_df.loc[:, 'group']

# This is where masks for adjacent rois are saved
MASK_LOADPATH = os.path.join(
    PTESTS_PATH, 'test_files', 'polygon_merger_roi_masks')
maskpaths = [
    os.path.join(MASK_LOADPATH, j) for j in os.listdir(MASK_LOADPATH)
    if j.endswith('.png')]

# %%===========================================================================
#
# =============================================================================


