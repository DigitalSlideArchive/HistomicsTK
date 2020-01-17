#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 01:38:16 2019.

@author: mtageld
"""
import girder_client
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
    monitoPrefix : str
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

def backup_annotation_jsons(
        gc, folderid, local, save_json=True,
        callback=None, callback_kwargs=dict()):
    """Dump annotations of folder and subfolders locally recursively.

    This reproduces this tiered structure locally and (possibly) dumps
    annotations there. Adapted from Lee A.D. Cooper

    Parameters
    -----------
    gc : girder_client.GirderClient
        authenticated girder client instance

    folderid : str
        girder id of source (base) folder

    local : str
        local path to dump annotations

    save_json : bool
        whether to dump annotations as json file

    callback : function
        function to call that takes in AT LEAST the following params
        - annotations: loaded annotations
        - local: local directory

    callback_kwargs : dict
        kwargs to pass along to callback

    """
    print("\n== Dumping results to:", local, " ==\n")

    # pull annotations for each slide in folder 'id'
    all_items = gc.listItem(folderId=folderid, limit=None)
    for itemidx, item in enumerate(all_items):

        monitorStr = "Item %d: %s" % (itemidx, item['name'])

        try:
            # pull annotation
            print("%s: load annotations" % monitorStr)
            annotations = gc.get('/annotation/item/' + item['_id'])

            if annotations is not None:

                # dump to JSON in local folder
                if save_json:
                    print("%s: save json" % monitorStr)
                    savepath = os.path.join(local, item['name'] + '.json')
                    with open(savepath, 'w') as fout:
                        json.dump(annotations, fout)

                # run callback
                if callback is not None:
                    print("%s: run callback" % monitorStr)
                    callback(
                        annotations=annotations, local=local,
                        **callback_kwargs)

        except Exception as e:
            print(str(e))

    # for each folder, create a new folder in 'local' and call self
    for folder in gc.listFolder(parentId=folderid):

        # create folder in local
        new_folder = os.path.join(local, folder['name'])
        os.mkdir(new_folder)

        # call self
        backup_annotation_jsons(
            gc, folder['_id'], new_folder,
            callback=callback, callback_kwargs=callback_kwargs)