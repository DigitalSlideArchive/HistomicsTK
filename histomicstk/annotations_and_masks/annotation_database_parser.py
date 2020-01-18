# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:19:18 2019

@author: tageldim
"""
import os
from histomicstk.workflows.specific_workflows import dump_annotations_workflow
from histomicstk.workflows.workflow_runner import (
    Workflow_runner, Slide_iterator)


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
        - annotations: loaded annotations
        - local: local directory

    callback_kwargs : dict
        kwargs to pass along to callback

    """
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
        monitorPrefix=local)

    workflow_runner.run()

    # for each folder, create a new folder in 'local' and call self
    for folder in gc.listFolder(parentId=folderid):

        # create folder in local
        new_folder = os.path.join(local, folder['name'])
        os.mkdir(new_folder)

        # call self
        dump_annotations_locally(
            gc, folder['_id'], new_folder,
            callback=callback, callback_kwargs=callback_kwargs)
