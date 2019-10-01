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


if __name__ == '__main__':
    unittest.main()




# %%===========================================================================
# %%===========================================================================
# %%===========================================================================


from histomicstk.saliency.cellularity_detection import (
    Cellularity_detector_superpixels)
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    delete_annotations_in_slide)

# from the TCGA-A2-A3XS-DX1 ROI in Amgad et al, 2019
cnorm_main = {
    'mu': np.array([8.74108109, -0.12440419,  0.0444982]),
    'sigma': np.array([0.6135447, 0.10989545, 0.0286032]),
}

cds = Cellularity_detector_superpixels(
    gc, slide_id='',
    MAG=3.0, compactness=0.1, spixel_size_baseMag=256 * 256,
    max_cellularity=40,
    visualize_spixels=False, visualize_contiguous=True,
    get_tissue_mask_kwargs={
        'deconvolve_first': False,
        'n_thresholding_steps': 2,
        'sigma': 1.5,
        'min_size': 500, },
    verbose=2, monitorPrefix='test',
    logging_savepath=logging_savepath)
cds.set_color_normalization_values(
    mu=cnorm_main['mu'], sigma=cnorm_main['sigma'], what='main')

# %%===========================================================================


def cellularity_detection_workflow_single_slide(
        gc, cds, slide_id, destination_folder_id=None,
        keep_existing_annotations=False):
    """Run Cellularity_detector_superpixels for single slide."""

    # copy slide to target folder, otherwise work in-place
    if destination_folder_id is not None:
        resp = gc.post(
            "/item/%s/copy?folderId=%s&copyAnnotations=%s" %
            (slide_id, destination_folder_id, keep_existing_annotations))
        slide_id = resp['_id']

    elif not keep_existing_annotations:
        delete_annotations_in_slide(gc, slide_id)

    # run cds for this slide
    cds.slide_id = slide_id
    cds.run()


# %%===========================================================================

from histomicstk.utils.general_utils import Base_HTK_Class

# Params for workflow runner
destination_folder_id = "5d9246f6bd4404c6b1faaa89"
slide_iterator = Slide_iterator(gc, source_folder_id=SAMPLE_SOURCE_FOLDER_ID)
workflow = cellularity_detection_workflow_single_slide
workflow_kwargs = {
    'gc': gc, 'cds': cds,
}

# %%===========================================================================

si = slide_iterator.run()

# %%===========================================================================

slide_info = next(si)


# %%===========================================================================
