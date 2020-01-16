# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:19:18 2019

@author: tageldim
"""
import girder_client
import json
import os


def connect_to_api(apiurl, apikey=None, interactive=True):
    """Connect to a specific girder API"""
    assert interactive or (apikey is not None)
    gc = girder_client.GirderClient(apiUrl=apiurl)
    if apikey is not None:
        interactive = False
    if interactive:
        gc.authenticate(interactive=True)
    else:
        gc.authenticate(apiKey=apikey)
    return gc


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
