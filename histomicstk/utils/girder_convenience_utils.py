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
    gc : gider_client.GirderClient
        authenticated girder client instance
    annotation_id : str
        girder id of annotation
    groups_to_add : list
        each entry is a dict containing the information about user groups
        to add and their permission levels. A sample entry must have the
        following keys
        - level, int -> 0 (view), 1 (edit) or 2 (owner)
        - name, str -> name of user group
        - id, st -> girder id of user group
    replace_original_groups : bool
        whether to replace original groups or append to them
    users_to_add : list
        each entry is a dict containing the information about user
        to add and their permission levels. A sample entry must have the
        following keys
        - level, int -> 0 (view), 1 (edit) or 2 (owner)
        - login, str -> username of user
        - id, st -> girder id of user
    replace_original_users
        whether to replace original users or append to them

    Returns
    -------
    dict
        server response

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
    gc : girder_client.GirderClient
        authenticated girder client
    slide_id : str
        girder id of slide
    monitorPrefix : str
        prefix to prepend to printed statements
    kwargs
        passed as-is to update_permissions_for_annotation()

    Returns
    -------
    list
        each entry is a dict of the server response.

    """
    # get annotations for slide
    slide_annotations = gc.get('/annotation/item/' + slide_id)

    resps = []
    for annidx, ann in enumerate(slide_annotations):
        print("%s: annotation %d of %d" % (
            monitorPrefix, annidx, len(slide_annotations)))
        resp = update_permissions_for_annotation(
            gc=gc, annotation_id=ann['_id'], **kwargs)
        resps.append(resp)
    return resps


def update_permissions_for_annotations_in_folder(
        gc, folderid, workflow_kwargs, monitor='', verbose=True):
    """Update permissions for all annotations in a folder.

    Parameters
    ----------
    gc : girder_client.GirderClient
        authenticated girder client
    folderid : str
        girder id of folder
    workflow_kwargs : dict
        kwargs to pass to update_permissions_for_annotations_in_slide()
    monitor : str
        text to prepend to printed statements
    verbose : bool
        print statements to screen?

    Returns
    -------
    None

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


def update_styles_for_annotation(gc, ann, changes):
    """Update styles for all relevant elements in an annotation.

    Parameters
    ----------
    gc : girder_client.GirderClient
        authenticated girder client
    ann : dict
        annotation
    changes : dict
        indexed by current group name to be updated, and values are
        the new styles. Each element in ann["annotation"]["elements"]
        whose current "group" attribute is in this dict's keys is
        updated according to the new style.

    Returns
    -------
    dict
        server response

    """
    # find out if annotation needs editing
    if 'groups' not in ann.keys():
        return
    elif not any([g in changes.keys() for g in ann['groups']]):
        return

    # edit elements one by one
    for el in ann['annotation']['elements']:
        if el['group'] in changes.keys():
            el.update(changes[el['group']])
    print("  updating ...")
    return gc.put("/annotation/%s" % ann['_id'], json=ann['annotation'])


def update_styles_for_annotations_in_slide(
        gc, slide_id, monitorPrefix='', callback=None, **kwargs):
    """Update styles for all annotations in a slide.

    Parameters
    ----------
    gc : girder_client.GirderClient
        authenticated girder client
    slide_id : str
        girder id of slide
    monitorPrefix : str
        prefix to prepend to printed statements
    callback : function
        if None, update_styles_for_annotation() is used. Must be able to
        accept the parameters
        - gc - authenticated girder client
        - ann - dict, annotation
        and must return the a dictionary.
    kwargs
        passed as-is to the callback

    Returns
    -------
    list
        each entry is a dict of the server response.

    """
    # get annotations for slide
    slide_annotations = gc.get('/annotation/item/' + slide_id)

    if callback is None:
        callback = update_styles_for_annotation

    resps = []
    for annidx, ann in enumerate(slide_annotations):
        print("%s: annotation %d of %d" % (
            monitorPrefix, annidx, len(slide_annotations)))
        resp = callback(gc=gc, ann=ann, **kwargs)
        resps.append(resp)
    return resps


def update_styles_for_annotations_in_folder(
        gc, folderid, workflow_kwargs, monitor='', verbose=True):
    """Update styles for all annotations in a slide.

    Parameters
    ----------
    gc : girder_client.GirderClient
        authenticated girder client
    folderid : str
        girder id of folder
    workflow_kwargs : dict
        kwargs to pass to Update styles for all annotations in a slide()
    monitor : str
        text to prepend to printed statements
    verbose : bool
        print statements to screen?

    Returns
    -------
    None

    """
    workflow_kwargs.update({'gc': gc})
    workflow_runner = Workflow_runner(
        slide_iterator=Slide_iterator(
            gc, source_folder_id=folderid,
            keep_slides=None,
        ),
        workflow=update_styles_for_annotations_in_slide,
        workflow_kwargs=workflow_kwargs,
        monitorPrefix=monitor,
        verbose=verbose,
    )
    workflow_runner.run()
