#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 18:12:48 2019

@author: mtageld
"""
import tempfile
import girder_client
import numpy as np

# %%===========================================================================
# Constants & prep work

APIURL = 'http://candygram.neurology.emory.edu:8080/api/v1/'
SAMPLE_SLIDE_ID = "5d586d76bd4404c6b1f286ae"
# SAMPLE_SLIDE_ID = "5d8c296cbd4404c6b1fa5572"

gc = girder_client.GirderClient(apiUrl=APIURL)
gc.authenticate(apiKey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')

logging_savepath = tempfile.mkdtemp()

# from the TCGA-A2-A3XS-DX1 ROI in Amgad et al, 2019
cnorm_main = {
    'mu': np.array([8.74108109, -0.12440419,  0.0444982]),
    'sigma': np.array([0.6135447, 0.10989545, 0.0286032]),
}

# %%===========================================================================

# Params for slide iterator
source_folder_id = "5d5c28c6bd4404c6b1f3d598"
keep_slides = None
discard_slides = None
# gc

# Params for workflow runner
# destin_folder_id = "5d9246f6bd4404c6b1faaa89"

# %%===========================================================================

