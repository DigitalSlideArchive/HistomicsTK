#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 18:12:48 2019

@author: mtageld
"""
import unittest
import tempfile
import girder_client
import numpy as np
from histomicstk.workflows.workflow_runner import Slide_iterator

# %%===========================================================================
# Constants & prep work

APIURL = 'http://candygram.neurology.emory.edu:8080/api/v1/'
SAMPLE_SOURCE_FOLDER_ID = "5d5c28c6bd4404c6b1f3d598"

gc = girder_client.GirderClient(apiUrl=APIURL)
gc.authenticate(apiKey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')

logging_savepath = tempfile.mkdtemp()

## from the TCGA-A2-A3XS-DX1 ROI in Amgad et al, 2019
#cnorm_main = {
#    'mu': np.array([8.74108109, -0.12440419,  0.0444982]),
#    'sigma': np.array([0.6135447, 0.10989545, 0.0286032]),
#}

# %%===========================================================================


class Slide_iterator_Test(unittest.TestCase):
    """Test slide iterator."""

    def test_Slide_iterator(self):
        """Test Slide_iterator.run()."""
        si = Slide_iterator(gc, source_folder_id=SAMPLE_SOURCE_FOLDER_ID)

        self.assertGreaterEqual(len(si.slide_ids), 1)

        sir = si.run()
        for i in range(2):
            slide_info = next(sir)

        self.assertTrue(all(
            [k in slide_info.keys() for k in
             ('name', '_id', 'levels', 'magnification', 'mm_x', 'mm_y',
              'sizeX', 'sizeY', 'tileHeight', 'tileWidth')]))


# %%===========================================================================


if __name__ == '__main__':
    unittest.main()




# %%===========================================================================
# %%===========================================================================
# %%===========================================================================

#from histomicstk.utils.general_utils import Base_HTK_Class
#
## Params for workflow runner
## destin_folder_id = "5d9246f6bd4404c6b1faaa89"
#
## %%===========================================================================
#
#
#
#
