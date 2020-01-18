#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 01:38:16 2019.

@author: mtageld
"""
import json
import os
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    delete_annotations_in_slide)


def cellularity_detection_workflow(
        gc, cdo, slide_id, monitorPrefix='',
        destination_folder_id=None, keep_existing_annotations=False):
    """Run cellularity detection for single slide.

    The cellularity detection algorithm can either be
    Cellularity_detector_superpixels or Cellularity_detector_thresholding.

    Arguments
    -----------
    gc : object
        girder client object
    cdo : object
        Cellularity_detector object instance. Can either be
        Cellularity_detector_superpixels() or
        Cellularity_detector_thresholding(). The thresholding-based workflow
        seems to be more robust, despite being simpler.
    slide_id : str
        girder id of slide on which workflow is done
    monitorPrefix : str
        this will set the cds monitorPrefix attribute
    destination_folder_id : str or None
        if not None, copy slide to this girder folder and post results
        there instead of original slide.
    keep_existing_annotations : bool
        keep existing annotations in slide when posting results?

    """
    cdo.monitorPrefix = monitorPrefix

    # copy slide to target folder, otherwise work in-place
    if destination_folder_id is not None:
        cdo._print1("%s: copying slide to destination folder" % monitorPrefix)
        resp = gc.post(
            "/item/%s/copy?folderId=%s&copyAnnotations=%s" %
            (slide_id, destination_folder_id, keep_existing_annotations))
        slide_id = resp['_id']

    elif not keep_existing_annotations:
        cdo._print1("%s: deleting existing annotations" % monitorPrefix)
        delete_annotations_in_slide(gc, slide_id)

    # run cds for this slide
    cdo.slide_id = slide_id
    cdo.run()

    return slide_id

# %%===========================================================================


def dump_annotations_workflow(
        gc, slide_id, local, save_json=True, monitorPrefix='',
        callback=None, callback_kwargs=dict()):
    """Dump annotations for single slide into the local folder.

    Parameters
    -----------
    gc : girder_client.GirderClient
        authenticated girder client instance

    slide_id : str
        girder id of item (slide)

    monitorPrefix : str
        prefix to monitor string

    local : str
        local path to dump annotations

    save_json : bool
        whether to dump annotations as json file

    callback : function
        function to call that takes in AT LEAST the following params
        - item: girder response with item information
        - annotations: loaded annotations
        - local: local directory
        - monitorPrefix: string

    callback_kwargs : dict
        kwargs to pass along to callback

    """
    try:
        item = gc.get('/item/%s' % slide_id)

        savepath_base = os.path.join(local, item['name'])

        # dump item information json
        if save_json:
            print("%s: save item info" % monitorPrefix)
        with open(savepath_base + '.json', 'w') as fout:
            json.dump(item, fout)

        # pull annotation
        print("%s: load annotations" % monitorPrefix)
        annotations = gc.get('/annotation/item/' + item['_id'])

        if annotations is not None:

            # dump annotations to JSON in local folder
            if save_json:
                print("%s: save annotations" % monitorPrefix)
                with open(savepath_base + '_annotations.json', 'w') as fout:
                    json.dump(annotations, fout)

            # run callback
            if callback is not None:
                print("%s: run callback" % monitorPrefix)
                callback(
                    item=item, annotations=annotations, local=local,
                    monitorPrefix=monitorPrefix, **callback_kwargs)

    except Exception as e:
        print(str(e))

# %%===========================================================================
