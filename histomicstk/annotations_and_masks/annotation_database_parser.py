"""
Created on Thu Dec 12 13:19:18 2019

@author: tageldim
"""
import copy
import json
import os

from sqlalchemy import create_engine
from sqlalchemy.types import Boolean, Integer, String

from histomicstk.annotations_and_masks.annotation_and_mask_utils import \
    parse_slide_annotations_into_tables
from histomicstk.utils.girder_convenience_utils import \
    get_absolute_girder_folderpath
from histomicstk.workflows.workflow_runner import (Slide_iterator,
                                                   Workflow_runner)

# Helper functions


def _add_item_to_sqlite(dbcon, item):
    from pandas import DataFrame

    # modify item info to prep for appending to sqlite table
    item_info = copy.deepcopy(item)
    item_info['largeImage'] = str(item_info['largeImage'])

    item_info_dtypes = {
        '_id': String(),
        '_modelType': String(),
        'baseParentId': String(),
        'baseParentType': String(),
        'copyOfItem': String(),
        'created': String(),
        'creatorId': String(),
        'description': String(),
        'folderId': String(),
        'largeImage': String(),
        'name': String(),
        'size': Integer(),
        'updated': String(),
    }

    # in case anything is not in the schema, drop it
    item_info = {
        k: v for k, v in item_info.items()
        if k in item_info_dtypes.keys()}

    # convert to df and add to items table
    item_info_df = DataFrame.from_dict(item_info, orient='index').T
    item_info_df.to_sql(
        name='items', con=dbcon, if_exists='append',
        dtype=item_info_dtypes, index=False)


def _add_folder_to_sqlite(dbcon, folder_info):
    from pandas import DataFrame

    # modify folder info to prep for appending to sqlite table
    folder_info_dtypes = {
        '_accessLevel': Integer(),
        '_id': String(),
        '_modelType': String(),
        'baseParentId': String(),
        'baseParentType': String(),
        'created': String(),
        'creatorId': String(),
        'description': String(),
        'name': String(),
        'parentCollection': String(),
        'parentId': String(),
        'public': Boolean(),
        'size': Integer(),
        'updated': String(),
        'folder_path': String(),
    }

    # in case anything is not in the schema, drop it
    folder_info = {
        k: v for k, v in folder_info.items()
        if k in folder_info_dtypes.keys()}

    # convert to df and add to items table
    folder_info_df = DataFrame.from_dict(folder_info, orient='index').T
    folder_info_df.to_sql(
        name='folders', con=dbcon, if_exists='append',
        dtype=folder_info_dtypes, index=False)


def _add_annotation_docs_to_sqlite(dbcon, annotation_docs, item):
    # add full item path for convenience
    annotation_docs.loc[:, 'item_name'] = item['name']

    # save tables to sqlite
    annotation_docs.to_sql(
        name='annotation_docs', con=dbcon, if_exists='append',
        dtype={
            'annotation_girder_id': String(),
            '_modelType': String(),
            '_version': Integer(),
            'itemId': String(),
            'item_name': String(),
            'created': String(),
            'creatorId': String(),
            'public': Boolean(),
            'updated': String(),
            'updatedId': String(),
            'groups': String(),
            'element_count': Integer(),
            'element_details': Integer()},
        index=False,
    )


def _add_annotation_elements_to_sqlite(dbcon, annotation_elements):
    # drop index relative to JSON since its pretty arbitrary and would
    # change if the same girder client was used to get annotations twice
    # the actual girder ID string is what really matters and should be used
    annotation_elements.drop(
        labels=['annidx', 'elementidx'], axis=1, inplace=True)

    annotation_elements.to_sql(
        name='annotation_elements', con=dbcon, if_exists='append',
        dtype={
            'annotation_girder_id': String(),
            'element_girder_id': String(),
            'type': String(),
            'group': String(),
            'label': String(),
            'color': String(),
            'xmin': Integer(),
            'xmax': Integer(),
            'ymin': Integer(),
            'ymax': Integer(),
            'bbox_area': Integer(),
            'coords_x': String(),
            'coords_y': String()},
        index=False,
    )


def parse_annotations_to_local_tables(
        item, annotations, local, monitorPrefix='',
        save_csv=True, save_sqlite=False, dbcon=None):
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

    save_sqlite : bool
        whether to save the backup into an sqlite database

    dbcon : sqlalchemy.create_engine.connect() object
        IGNORE THIS PARAMETER!! This is used internally.

    monitorPrefix : str
        text to prepend to printed statements

    """
    print('%s: parse to tables' % monitorPrefix)
    savepath_base = os.path.join(local, item['name'])
    annotation_docs, annotation_elements = \
        parse_slide_annotations_into_tables(annotations)

    if save_csv:
        annotation_docs.to_csv(savepath_base + '_docs.csv')
        annotation_elements.to_csv(savepath_base + '_elements.csv')

    if save_sqlite:
        assert dbcon is not None, 'You must connect to database first!'
        _add_annotation_docs_to_sqlite(dbcon, annotation_docs, item)
        _add_annotation_elements_to_sqlite(dbcon, annotation_elements)


# Workflow at a single slide level


def dump_annotations_workflow(
        gc, slide_id, local, monitorPrefix='',
        save_json=True, save_sqlite=False, dbcon=None,
        callback=None, callback_kwargs=None):
    """Dump annotations for single slide into the local folder.

    Parameters
    ----------
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

    save_sqlite : bool
        whether to save the backup into an sqlite database

    dbcon : sqlalchemy.create_engine.connect() object
        IGNORE THIS PARAMETER!! This is used internally.

    callback : function
        function to call that takes in AT LEAST the following params
        - item: girder response with item information
        - annotations: loaded annotations
        - local: local directory
        - monitorPrefix: string

    callback_kwargs : dict
        kwargs to pass along to callback

    """
    callback_kwargs = callback_kwargs or {}
    try:
        item = gc.get('/item/%s' % slide_id)

        savepath_base = os.path.join(local, item['name'])

        # dump item information json
        if save_json:
            print('%s: save item info' % monitorPrefix)
            with open(savepath_base + '.json', 'w') as fout:
                json.dump(item, fout)

        # save folder info to sqlite
        if save_sqlite:
            _add_item_to_sqlite(dbcon, item)

        # pull annotation
        print('%s: load annotations' % monitorPrefix)
        annotations = gc.get('/annotation/item/' + item['_id'])

        if annotations is not None:

            # dump annotations to JSON in local folder
            if save_json:
                print('%s: save annotations' % monitorPrefix)
                with open(savepath_base + '_annotations.json', 'w') as fout:
                    json.dump(annotations, fout)

            # run callback
            if callback is not None:
                print('%s: run callback' % monitorPrefix)
                callback(
                    item=item, annotations=annotations, local=local,
                    dbcon=dbcon, monitorPrefix=monitorPrefix,
                    **callback_kwargs)

    except Exception as e:
        print(str(e))


# Main method


def dump_annotations_locally(
        gc, folderid, local, save_json=True,
        save_sqlite=False, dbcon=None,
        callback=None, callback_kwargs=None):
    """Dump annotations of folder and subfolders locally recursively.

    This reproduces this tiered structure locally and (possibly) dumps
    annotations there. Adapted from Lee A.D. Cooper

    Parameters
    ----------
    gc : girder_client.GirderClient
        authenticated girder client instance

    folderid : str
        girder id of source (base) folder

    local : str
        local path to dump annotations

    save_json : bool
        whether to dump annotations as json file

    save_sqlite : bool
        whether to save the backup into an sqlite database

    dbcon : sqlalchemy.create_engine.connect() object
        IGNORE THIS PARAMETER!! This is used internally.

    callback : function
        function to call that CAN accept AT LEAST the following params
        - item: girder response with item information
        - annotations: loaded annotations
        - local: local directory
        - monitorPrefix: string
        - dbcon: sqlalchemy.create_engine.connect() object
        You can just add kwargs at the end of your callback definition
        for simplicity.

    callback_kwargs : dict
        kwargs to pass along to callback. DO NOT pass any of the parameters
        item, annotations, local, monitorPrefix, or dbcon as these will be
        internally passed. Just include any specific parameters for the
        callback. See parse_annotations_to_local_tables() above for
        an example of a callback and the unir test of this function.

    """
    callback_kwargs = callback_kwargs or {}
    assert save_json or save_sqlite, 'must save results somehow!'
    monitor = os.path.basename(local)

    # get folder info
    folder_info = gc.get('folder/%s' % folderid)
    folder_info['folder_path'] = get_absolute_girder_folderpath(
        gc=gc, folder_info=folder_info)

    # connect to sqlite database -- only first stack does this
    if save_sqlite and (dbcon is None):
        db_path = os.path.join(local, folder_info['name'] + '.sqlite')
        sql_engine = create_engine('sqlite:///' + db_path, echo=False)
        dbcon = sql_engine.connect()

    # save folder information json
    if save_json:
        print('%s: save folder info' % monitor)
        savepath = os.path.join(local, folder_info['name'] + '.json')
        with open(savepath, 'w') as fout:
            json.dump(folder_info, fout)

    # save folder info to sqlite
    if save_sqlite:
        _add_folder_to_sqlite(dbcon, folder_info)

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
            'save_sqlite': save_sqlite,
            'dbcon': dbcon,
            'callback': callback,
            'callback_kwargs': callback_kwargs,
        },
        monitorPrefix=monitor)

    workflow_runner.run()

    # for each subfolder, create a new folder locally and call self
    for folder in gc.listFolder(parentId=folderid):

        # create folder in local
        new_folder = os.path.join(local, folder['name'])
        os.mkdir(new_folder)

        # call self with same parameters
        dump_annotations_locally(
            gc=gc, folderid=folder['_id'], local=new_folder,
            save_json=save_json, save_sqlite=save_sqlite, dbcon=dbcon,
            callback=callback, callback_kwargs=callback_kwargs)
