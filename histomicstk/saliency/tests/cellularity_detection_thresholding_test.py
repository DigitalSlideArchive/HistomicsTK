#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 09:47:52 2019.

@author: mtageld
"""
import os
import unittest
import tempfile
import shutil
import girder_client
# import numpy as np
from pandas import read_csv
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    delete_annotations_in_slide)
from histomicstk.saliency.cellularity_detection_thresholding import (
    Cellularity_detector_thresholding)

# %%===========================================================================
# Constants & prep work

APIURL = 'http://candygram.neurology.emory.edu:8080/api/v1/'
# SAMPLE_SLIDE_ID = "5d586d76bd4404c6b1f286ae"
SAMPLE_SLIDE_ID = "5d8c296cbd4404c6b1fa5572"
# SAMPLE_SLIDE_ID = "5d94ee48bd4404c6b1fb0b40"

gc = girder_client.GirderClient(apiUrl=APIURL)
gc.authenticate(apiKey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')

# read GT codes dataframe
GTcodes = read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'saliency_GTcodes.csv'))

logging_savepath = tempfile.mkdtemp()

# %%===========================================================================


class CellularityDetectionThresholdingTest(unittest.TestCase):
    """Test methods for getting cellularity by thresholding."""

    def test_cellularity_detection_thresholding(self):
        """Test Cellularity_detector_thresholding()."""
        # deleting existing annotations in target slide (if any)
        delete_annotations_in_slide(gc, SAMPLE_SLIDE_ID)

        # run cellularity detector
        cdt = Cellularity_detector_thresholding(
            gc, slide_id=SAMPLE_SLIDE_ID, GTcodes=GTcodes,
            verbose=2, monitorPrefix='test',
            logging_savepath=logging_savepath)
        tissue_pieces = cdt.run()

        # check
        self.assertEqual(len(tissue_pieces), 1)
        self.assertTrue(all(
            [j in tissue_pieces[0].__dict__.keys() for j in
             ('labeled', 'ymin', 'xmin', 'ymax', 'xmax')]))

        # cleanup
        shutil.rmtree(logging_savepath)

# %%===========================================================================


if __name__ == '__main__':
    unittest.main()
