# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 22:50:03 2019

@author: tageldim
"""

import unittest

import numpy as np
import girder_client

from histomicstk.utils.polygon_and_mask_utils import (
        get_image_from_htk_response,)

# %%===========================================================================
# Constants & prep work
# =============================================================================

APIURL = 'http://demo.kitware.com/histomicstk/api/v1/'
SOURCE_FOLDER_ID = '5bbdeba3e629140048d017bb'
SAMPLE_SLIDE_ID = "5bbdeed1e629140048d01bcb"

gc= girder_client.GirderClient(apiUrl = APIURL)
gc.authenticate(interactive=True)

# %%===========================================================================
# GirderUtilsTest
# =============================================================================

class GirderUtilsTest(unittest.TestCase):

    def test_get_image_from_htk_response(self):
        
        getStr = "/item/%s/tiles/region?left=%d&right=%d&top=%d&bottom=%d" % (
          SAMPLE_SLIDE_ID, 45000, 45100, 25000, 25100)
        resp  = gc.get(getStr, jsonResp=False)
        rgb = get_image_from_htk_response(resp)

        unittest.TestCase.assertTupleEqual(rgb.shape, (100, 100, 3))
        
# %%===========================================================================
# 
# =============================================================================
