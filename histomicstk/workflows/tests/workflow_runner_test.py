#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 18:12:48 2019.

@author: mtageld
"""
import unittest
import shutil
import tempfile
import girder_client
import numpy as np
from histomicstk.workflows.workflow_runner import Slide_iterator
from histomicstk.saliency.cellularity_detection import (
    Cellularity_detector_superpixels)
from histomicstk.workflows.workflow_runner import Workflow_runner
from histomicstk.workflows.specific_workflows import (
    cellularity_detection_workflow)

# %%===========================================================================
# Constants & prep work

APIURL = 'http://candygram.neurology.emory.edu:8080/api/v1/'
SAMPLE_SOURCE_FOLDER_ID = "5d5c28c6bd4404c6b1f3d598"
SAMPLE_DESTINATION_FOLDER_ID = "5d9246f6bd4404c6b1faaa89"

# color norm. standard (from TCGA-A2-A3XS-DX1, Amgad et al, 2019)
cnorm_main = {
    'mu': np.array([8.74108109, -0.12440419,  0.0444982]),
    'sigma': np.array([0.6135447, 0.10989545, 0.0286032]),
}

# girder client
gc = girder_client.GirderClient(apiUrl=APIURL)
# gc.authenticate(interactive=True)
gc.authenticate(apiKey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')

logging_savepath = tempfile.mkdtemp()

# Cellularity_detector_superpixels parameters
cds_params = {
    'gc': gc, 'slide_id': '',
    'MAG': 3.0, 'compactness': 0.1, 'spixel_size_baseMag': 256 * 256,
    'max_cellularity': 40,
    'visualize_spixels': False, 'visualize_contiguous': True,
    'get_tissue_mask_kwargs': {
        'deconvolve_first': False,
        'n_thresholding_steps': 2,
        'sigma': 1.5,
        'min_size': 500, },
    'verbose': 2, 'logging_savepath': logging_savepath,
}

# %%===========================================================================


class Slide_iterator_Test(unittest.TestCase):
    """Test slide iterator."""

    def test_Slide_iterator(self):
        """Test Slide_iterator.run()."""
        slide_iterator = Slide_iterator(
            gc, source_folder_id=SAMPLE_SOURCE_FOLDER_ID)

        self.assertGreaterEqual(len(slide_iterator.slide_ids), 1)

        si = slide_iterator.run()
        for i in range(2):
            slide_info = next(si)

        self.assertTrue(all(
            [k in slide_info.keys() for k in
             ('name', '_id', 'levels', 'magnification', 'mm_x', 'mm_y',
              'sizeX', 'sizeY', 'tileHeight', 'tileWidth')]))


# %%===========================================================================


class Workflow_runner_Test(unittest.TestCase):
    """Test slide iterator."""

    def test_workflow_runner_using_cellularity_detector(self):
        """Test workflow runner for cellularity detection."""

        # Init Cellularity_detector_superpixels
        cds = Cellularity_detector_superpixels(**cds_params)
        cds.set_color_normalization_values(
            mu=cnorm_main['mu'], sigma=cnorm_main['sigma'], what='main')

        # Init workflow runner
        workflow_runner = Workflow_runner(
            slide_iterator=Slide_iterator(
                gc, source_folder_id=SAMPLE_SOURCE_FOLDER_ID,
                keep_slides=['TCGA-A1-A0SK-01Z-00-DX1_POST.svs', ]),
            workflow=cellularity_detection_workflow,
            workflow_kwargs={
                'gc': gc,
                'cds': cds,
                'destination_folder_id': SAMPLE_DESTINATION_FOLDER_ID,
                'keep_existing_annotations': False, },
            logger=cds.logger,
            monitorPrefix='test')

        # Now run
        workflow_runner.run()

# %%===========================================================================


if __name__ == '__main__':
    unittest.main()

    # cleanup
    shutil.rmtree(logging_savepath)
