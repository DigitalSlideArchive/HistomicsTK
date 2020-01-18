# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:19:18 2019

@author: tageldim
"""
import os
import json
from histomicstk.workflows.specific_workflows import dump_annotations_workflow
from histomicstk.workflows.workflow_runner import (
    Workflow_runner, Slide_iterator)
from histomicstk.utils.girder_convenience_utils import (
    get_absolute_girder_folderpath)
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    parse_slide_annotations_into_tables)


def parse_annotations_to_local_tables(
        item, annotations, local, save_csv=True, monitorPrefix=''):
    """Parse loaded annotations for slide into tables.

    Parameters
    ----------
    item : dict
        girder response with item information
    annotations : dict
        loaded annotations
    local : str
        local directory
    save_csv : bool
        whether to use histomicstk.annotations_and_masks.annotation_and_mask.
        parse_slide_annotations_into_tables() to get a tabular representation
        (including some simple calculations like bounding box) and save
        the output as two csv files, one representing the annotation documents
        and the other representing the actual annotation elements (polygons).
    monitorPrefix : str
        text to prepend to printed statements

    """
    print("%s: parse to tables" % monitorPrefix)
    savepath_base = os.path.join(local, item['name'])
    annotation_docs, annotation_elements = \
        parse_slide_annotations_into_tables(annotations)
    if save_csv:
        annotation_docs.to_csv(savepath_base + '_docs.csv')
        annotation_elements.to_csv(savepath_base + '_elements.csv')


def dump_annotations_locally(
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
        - item: girder response with item information
        - annotations: loaded annotations
        - local: local directory
        - monitorPrefix: string

    callback_kwargs : dict
        kwargs to pass along to callback. DO NOT pass any of the parameters
        item, annotations, local, or monitorPrefix here as these will be
        internally passed. Just include any specific paremeters for the
        callback. See parse_annotations_to_local_tables() above for
        an example of a callback and the unir test of this function.

    """
    monitor = os.path.basename(local)
    # pull annotations for each slide in folder
    workflow_runner = Workflow_runner(
        slide_iterator=Slide_iterator(
            gc, source_folder_id=folderid,
            keep_slides=None,
        ),
        workflow=dump_annotations_workflow,
        workflow_kwargs={
            'gc': gc,
            'local': local,
            'save_json': save_json,
            'callback': callback,
            'callback_kwargs': callback_kwargs,
        },
        monitorPrefix=monitor)

    workflow_runner.run()

    # dump folder information json
    folder_info = gc.get("folder/%s" % folderid)
    folder_info['folder_path'] = get_absolute_girder_folderpath(
        gc=gc, folder_info=folder_info)
    if save_json:
        print("%s: save folder info" % monitor)
        savepath = os.path.join(local, folder_info['name'] + '.json')
        with open(savepath, 'w') as fout:
            json.dump(folder_info, fout)

    # for each subfolder, create a new folder locally and call self
    for folder in gc.listFolder(parentId=folderid):

        # create folder in local
        new_folder = os.path.join(local, folder['name'])
        os.mkdir(new_folder)

        # call self with same prameters
        dump_annotations_locally(
            gc=gc, folderid=folder['_id'], local=new_folder,
            save_json=save_json,
            callback=callback, callback_kwargs=callback_kwargs)
