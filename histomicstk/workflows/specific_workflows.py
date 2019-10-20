#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 01:38:16 2019.

@author: mtageld
"""
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    delete_annotations_in_slide)

# %%===========================================================================


def cellularity_detection_superpixels_workflow(
        gc, cds, slide_id, monitorPrefix='',
        destination_folder_id=None, keep_existing_annotations=False):
    """Run Cellularity_detector_superpixels for single slide.

    Arguments
    -----------
    gc : object
        girder client object
    cds : object
        Cellularity_detector_superpixels object instance
    slide_id : str
        girder id of slide on which workflow is done
    monitoPrefix : str
        this will set the cds monitorPrefix attribute
    destination_folder_id : str or None
        if not None, copy slide to this girder folder and post results
        there instead of original slide.
    keep_existing_annotations : bool
        keep existing annotations in slide when posting results?

    """
    cds.monitorPrefix = monitorPrefix

    # copy slide to target folder, otherwise work in-place
    if destination_folder_id is not None:
        cds._print1("%s: copying slide to destination folder" % monitorPrefix)
        resp = gc.post(
            "/item/%s/copy?folderId=%s&copyAnnotations=%s" %
            (slide_id, destination_folder_id, keep_existing_annotations))
        slide_id = resp['_id']

    elif not keep_existing_annotations:
        cds._print1("%s: deleting existing annotations" % monitorPrefix)
        delete_annotations_in_slide(gc, slide_id)

    # run cds for this slide
    cds.slide_id = slide_id
    cds.run()

    return slide_id

# %%===========================================================================
