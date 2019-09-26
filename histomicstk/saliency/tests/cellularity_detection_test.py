#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 02:25:34 2019.

@author: mtageld
"""

import girder_client
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    delete_annotations_in_slide)
from histomicstk.saliency.cellularity_detection import (
    Cellularity_detector_superpixels)

# %%===========================================================================
# Constants & prep work

APIURL = 'http://candygram.neurology.emory.edu:8080/api/v1/'
SAMPLE_SLIDE_ID = "5d586d76bd4404c6b1f286ae"

gc = girder_client.GirderClient(apiUrl=APIURL)
gc.authenticate(apiKey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')

# %%===========================================================================

# deleting existing annotations in target slide (if any)
delete_annotations_in_slide(gc, SAMPLE_SLIDE_ID)

# run cellularity detector
cds = Cellularity_detector_superpixels(
    gc, SAMPLE_SLIDE_ID, suppress_warnings=True, verbose=2,
    visualize_spixels=False, visualize_contiguous=True)
cds.run()
