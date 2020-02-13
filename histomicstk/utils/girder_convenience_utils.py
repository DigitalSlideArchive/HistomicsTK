# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:19:18 2019

@author: tageldim
"""
import girder_client
import json
from histomicstk.workflows.workflow_runner import Workflow_runner, \
    Slide_iterator


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


def get_absolute_girder_folderpath(gc, folder_id=None, folder_info=None):
    """Get absolute path for a girder folder."""
    assert any([j is not None for j in (folder_id, folder_info)])
    if folder_id is not None:
        folder_info = gc.get('/folder/%s' % folder_id)
    fpath = gc.get('/folder/%s/rootpath' % folder_info['_id'])
    fpath = "/".join(
        [j['object']['name'] for j in fpath]
    ) + "/" + folder_info['name'] + "/"
    return fpath


def update_permissions_for_annotation(
        gc, annotation_id,
        groups_to_add=[], replace_original_groups=False,
        users_to_add=[], replace_original_users=False):
    """Update permissions for a single annotation.

    Parameters
    ----------
    gc
    annotation_id
    groups_to_add
    replace_original_groups
    users_to_add
    replace_original_users

    Returns
    -------

    """
    # get current permissions
    current = gc.get('/annotation/%s/access' % annotation_id)

    # add or replace as needed
    if replace_original_groups:
        current['groups'] = []

    if replace_original_users:
        current['users'] = []

    for group in groups_to_add:
        current['groups'].append(group)

    for user in users_to_add:
        current['users'].append(user)

    # now update accordingly
    return gc.put('/annotation/%s/access?access=%s' % (
        annotation_id, json.dumps(current)))


def update_permissions_for_annotations_in_slide(
        gc, slide_id, monitorPrefix='', **kwargs):
    """Update permissions for all annotations in a slide.

    Parameters
    ----------
    gc
    slide_id
    monitorPrefix

    Returns
    -------

    """
    # get annotations for slide
    slide_annotations = gc.get('/annotation/item/' + slide_id)

    for annidx, ann in enumerate(slide_annotations):
        print("%s: annotation %d of %d" % (
            monitorPrefix, annidx, len(slide_annotations)))
        _ = update_permissions_for_annotation(
            gc=gc, annotation_id=ann['_id'], **kwargs)


def update_permissions_for_annotations_in_folder(
        gc, folderid, workflow_kwargs, monitor='', verbose=True):
    """Update permissions for all annotations in a folder.

    Parameters
    ----------
    gc
    folderid
    workflow_kwargs
    monitor

    Returns
    -------

    """
    workflow_kwargs.update({'gc': gc})
    workflow_runner = Workflow_runner(
        slide_iterator=Slide_iterator(
            gc, source_folder_id=folderid,
            keep_slides=None,
        ),
        workflow=update_permissions_for_annotations_in_slide,
        workflow_kwargs=workflow_kwargs,
        monitorPrefix=monitor,
        verbose=verbose,
    )
    workflow_runner.run()
