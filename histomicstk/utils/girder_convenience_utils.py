# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:19:18 2019

@author: tageldim
"""
# import os
import os

import girder_client
import json
from histomicstk.workflows.workflow_runner import Workflow_runner, \
    Slide_iterator, Annotation_iterator
import warnings
warnings.simplefilter('once', UserWarning)
ASKTOCONTINUE = True


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
        gc, annotation_id=None, annotation=None,
        groups_to_add=[], replace_original_groups=True,
        users_to_add=[], replace_original_users=True):
    """Update permissions for a single annotation.

    Parameters
    ----------
    gc : gider_client.GirderClient
        authenticated girder client instance
    annotation_id : str
        girder id of annotation
    annotation : dict
        overrides annotation_id if given
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
    # TODO -- this is an especially damaging bug! Fix me!
    GIRDERBUG = """
    We've discovered a bug that causes the dangerous side effect of 
    deleting annotation elements whent he access permissions are edited!
    This behavior happens CONSISTENTLY when we APPEND to the existing
    users of groups, and OCCASIONALLY when we replace the users or
    groups with new ones. Until this bug is fixed, try to avoid 
    changing annotation access permissions, and if you really do
    have to do it, make SURE you replace existing users/groups
    as opposed to appending to them."""
    warnings.warn(GIRDERBUG, RuntimeWarning)

    global ASKTOCONTINUE
    if ASKTOCONTINUE:
        warnings.warn("""
        > Press any key to continue at your own risk, or.. 
        > Press Control+C to stop.""")
        input("")
        ASKTOCONTINUE = False

    if annotation is not None:
        annotation_id = annotation['_id']
    elif annotation_id is None:
        raise Exception(
            "You must provide either the annotation or its girder id.")

    # get current permissions
    current = gc.get('/annotation/%s/access' % annotation_id)

    # add or replace as needed
    if replace_original_groups:
        current['groups'] = []
    else:
        raise Exception(GIRDERBUG)

    if replace_original_users:
        current['users'] = []
    else:
        raise Exception(GIRDERBUG)

    for group in groups_to_add:
        current['groups'].append(group)

    for user in users_to_add:
        current['users'].append(user)

    # now update accordingly
    # OLD WAY (BAD!!)
    # return gc.put('/annotation/%s/access?access=%s' % (
    #        annotation_id, json.dumps(current)))
    # PROPER WAY
    return gc.put('/annotation/%s/access' % annotation_id, data={
        'access': json.dumps(current)})


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
    anniter = Annotation_iterator(
        gc=gc, slide_id=slide_id,
        callback=update_permissions_for_annotation,
        callback_kwargs=kwargs,
        monitorPrefix=monitorPrefix)
    return anniter.apply_callback_to_all_annotations()


def update_permissions_for_annotations_in_folder(
        gc, folderid, workflow_kwargs, recursive=True,
        monitor='', verbose=True):
    """Update permissions for all annotations in a folder recursively.

    Parameters
    ----------
    gc : girder_client.GirderClient
        authenticated girder client
    folderid : str
        girder id of folder
    workflow_kwargs : dict
        kwargs to pass to update_permissions_for_annotations_in_slide()
    recursive : bool
        do this recursively for subfolders?
    monitor : str
        text to prepend to printed statements
    verbose : bool
        print statements to screen?

    Returns
    -------
    None

    """
    # update permissions for each slide in folder
    workflow_kwargs.update({'gc': gc})
    workflow_runner = Workflow_runner(
        slide_iterator=Slide_iterator(
            gc, source_folder_id=folderid,
            keep_slides=None,
        ),
        workflow=update_permissions_for_annotations_in_slide,
        workflow_kwargs=workflow_kwargs,
        recursive=recursive,
        monitorPrefix=monitor,
        verbose=verbose,
    )
    workflow_runner.run()


def update_styles_for_annotation(gc, annotation, changes):
    """Update styles for all relevant elements in an annotation.

    Parameters
    ----------
    gc : girder_client.GirderClient
        authenticated girder client
    annotation : dict
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
    if 'groups' not in annotation.keys():
        return
    elif not any([g in changes.keys() for g in annotation['groups']]):
        return

    # edit elements one by one
    for el in annotation['annotation']['elements']:
        if el['group'] in changes.keys():
            el.update(changes[el['group']])
    print("  updating ...")
    return gc.put(
        "/annotation/%s" % annotation['_id'], json=annotation['annotation'])


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
    kwargs
        passed as-is to the update_styles_for_annotation

    Returns
    -------
    list
        each entry is a dict of the server response.

    """
    if callback is None:
        callback = update_styles_for_annotation
    anniter = Annotation_iterator(
        gc=gc, slide_id=slide_id,
        callback=callback,
        callback_kwargs=kwargs,
        monitorPrefix=monitorPrefix)
    return anniter.apply_callback_to_all_annotations()


def update_styles_for_annotations_in_folder(
        gc, folderid, workflow_kwargs, recursive=True,
        monitor='', verbose=True):
    """Update styles for all annotations in a folder recursively.

    Parameters
    ----------
    gc : girder_client.GirderClient
        authenticated girder client
    folderid : str
        girder id of folder
    workflow_kwargs : dict
        kwargs to pass to Update styles for all annotations in a slide()
    recursive : bool
        do this recursively for subfolders?
    monitor : str
        text to prepend to printed statements
    verbose : bool
        print statements to screen?

    Returns
    -------
    None

    """
    # update annotation styles
    workflow_kwargs.update({'gc': gc})
    workflow_runner = Workflow_runner(
        slide_iterator=Slide_iterator(
            gc, source_folder_id=folderid,
            keep_slides=None,
        ),
        workflow=update_styles_for_annotations_in_slide,
        workflow_kwargs=workflow_kwargs,
        recursive=recursive,
        monitorPrefix=monitor,
        verbose=verbose,
    )
    workflow_runner.run()


def revert_annotation(
        gc, annotation_id=None, annotation=None, version=None,
        revert_to_nonempty_elements=False, only_revert_if_empty=True):
    """Revert an annotation to a previous version.

    Parameters
    ----------
    gc : girder_client.GirderClient
        authenticated girder client
    annotation_id : str
        girder id of annotation
    annotation : dict
        overrides annotation_id if given
    version : int
        versoin number for annotation. If None, and
        not revert_to_nonempty_elements
        the default behavior of the endpoint is evoked, which reverts the
        annotation if it was deleted and if not, reverts to the last version.
    revert_to_nonempty_elements : bool
        if true, reverts to the most recent version of the annotation
        with non-empty elements.

    Returns
    -------
    dict
        server response

    """
    if annotation is not None:
        annotation_id = annotation['_id']
    elif annotation_id is None:
        raise Exception(
            "You must provide either the annotation or its girder id.")

    history = gc.get("/annotation/%s/history" % annotation_id)

    # no need to revert if empty
    if only_revert_if_empty and len(history[0]["groups"]) > 0:
        return dict()

    if (version is None) and revert_to_nonempty_elements:

        # NOTE: even though the "history" may show
        # the elements as empty, the "groups" attribute is really the
        # indication if the annotation version actually has some elements.
        # TODO -- This is likely a bug (?); fix me!!!
        for ver in history:
            if len(ver["groups"]) > 0:
                version = ver['_version']
                break

    ver = "" if version is None else "?version=%d" % version

    if version is None:
        print("    Reverting ...")
    else:
        print("    Reverting to version %d" % version)

    return gc.put("/annotation/%s/history/revert%s" % (annotation_id, ver))


def revert_annotations_in_slide(
        gc, slide_id, monitorPrefix='', **kwargs):
    """Revert all annotations in a slide to a previous version.

    Parameters
    ----------
    gc : girder_client.GirderClient
        authenticated girder client
    slide_id : str
        girder id of slide
    monitorPrefix : str
        prefix to prepend to printed statements
    kwargs
        passed as-is to the revert_annotation

    Returns
    -------
    list
        each entry is a dict of the server response.

    """
    anniter = Annotation_iterator(
        gc=gc, slide_id=slide_id,
        callback=revert_annotation,
        callback_kwargs=kwargs,
        monitorPrefix=monitorPrefix)
    return anniter.apply_callback_to_all_annotations()


def revert_annotations_in_folder(
        gc, folderid, workflow_kwargs, recursive=True,
        monitor='', verbose=True):
    """Revert all annotations in a folder recursively.

    Parameters
    ----------
    gc : girder_client.GirderClient
        authenticated girder client
    folderid : str
        girder id of folder
    workflow_kwargs : dict
        kwargs to pass to revert_annotations_in_slide
    recursive : bool
        do this recursively for subfolders?
    monitor : str
        text to prepend to printed statements
    verbose : bool
        print statements to screen?

    Returns
    -------
    None

    """
    # update annotation styles
    workflow_kwargs.update({'gc': gc})
    workflow_runner = Workflow_runner(
        slide_iterator=Slide_iterator(
            gc, source_folder_id=folderid,
            keep_slides=None,
        ),
        workflow=revert_annotations_in_slide,
        workflow_kwargs=workflow_kwargs,
        recursive=recursive,
        monitorPrefix=monitor,
        verbose=verbose,
    )
    workflow_runner.run()

# %%===========================================================================


def reproduce_annotations_workflow(
        gc, folderid, annotation_jsonfile, monitorPrefix=''):
    """Dump annotations into single slide from local folder.

    Parameters
    -----------
    gc : girder_client.GirderClient
        authenticated girder client instance

    folderid : str
        girder id of girder folder to post reproduced annotations.

    annotation_jsonfile : str
        path to annotation json file

    monitorPrefix : str
        prefix to monitor string

    """
    try:
        # extract name + path
        itemname = os.path.basename(annotation_jsonfile).replace(
            '_annotations.json', '')
        local = os.path.dirname(annotation_jsonfile)

        # copy item without annotations
        with open(os.path.join(local, itemname + '.json'), 'r') as jf:
            source_item_info = json.load(jf)
        print("%s: copy item" % monitorPrefix)
        item = gc.post(
            '/item/%s/copy?folderId=%s&name=%s&copyAnnotations=False'
            % (source_item_info['_id'], folderid, itemname))

        # load annotations
        with open(annotation_jsonfile) as af:
            annotations = json.load(af)

        # now post
        n_annotations = len(annotations)
        for anno, annotation in enumerate(annotations):
            try:
                print("%s: post annotation %d of %d" % (
                    monitorPrefix, anno, n_annotations))
                _ = gc.post(
                    "/annotation?itemId=" + item['_id'],
                    json=annotation['annotation'])
            except Exception as e:
                print(e.__repr__())

    except Exception as e:
        print(e.__repr__())


def reproduce_annotations_from_backup(gc, folderid, local):
    """Reproduce annotations on HistomicsUI from local backup.

    This is the reverse functionality to dump_annotations.locally().
    This reproduces this tiered structure on HistomicsUI. The original
    slides (items) must still be there in the folder from the backup was
    made because these will be copied (without annotations) before the
    local annotations (from JSON files) are posted to them.

    Parameters
    -----------
    gc : girder_client.GirderClient
        authenticated girder client instance

    folderid : str
        girder id of girder folder to post reproduced annotations.

    local : str
        local path to get subfolders/slides/annotations

    """
    monitor = os.path.basename(local)

    # for each slide, copy it and post annotations
    jsonfiles = [
        os.path.join(local, j) for j in os.listdir(local)
        if j.endswith('_annotations.json')]
    for jsonfile in jsonfiles:
        reproduce_annotations_workflow(
            gc=gc, folderid=folderid, annotation_jsonfile=jsonfile,
            monitorPrefix=monitor)

    # for each subfolder, create a new folder on HistomicsUI and call self
    subdirs = [
        j for j in os.listdir(local) if os.path.isdir(os.path.join(local, j))]
    for subdir in subdirs:
        try:
            # create folder in HistomicsUI
            new_folder = gc.post('/folder?parentId=%s&name=%s' % (
                folderid, subdir))

            # call self with same prameters
            reproduce_annotations_from_backup(
                gc=gc, folderid=new_folder['_id'],
                local=os.path.join(local, subdir))

        except Exception as e:
            print(e.__repr__())


# %%===========================================================================
