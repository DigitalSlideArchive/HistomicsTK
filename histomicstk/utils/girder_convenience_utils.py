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


def backup_annotation_jsons(gc, folderid, local):
    """Dump annotations of folder and subfolders locally recursively.

    This reproduces this tiered structure locally and dumps annotations there.
    Adapted from Lee A.D. Cooper

    Parameters
    -----------
    gc : object
        girder client object

    folderid : str
        girder id of source (base) folder

    local : str
        local path to dump annotations

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

            # dump to JSON in local folder
            if annotations:
                print("%s: save json" % monitorStr)
                savepath = os.path.join(local, item['name'] + '.json')
                with open(savepath, 'w') as fout:
                    json.dump(annotations, fout)
        except Exception as e:
            print(str(e))

    # for each folder, create a new folder in 'local' and call self
    for folder in gc.listFolder(parentId=folderid):

        # create folder in local
        new_folder = os.path.join(local, folder['name'])
        os.mkdir(new_folder)

        # call self
        backup_annotation_jsons(gc, folder['_id'], new_folder)
