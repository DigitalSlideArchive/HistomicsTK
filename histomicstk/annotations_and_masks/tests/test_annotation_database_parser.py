import os
import shutil
import sys
import tempfile

import pandas as pd
import pytest
import sqlalchemy as db

from histomicstk.annotations_and_masks.annotation_database_parser import (
    dump_annotations_locally, parse_annotations_to_local_tables)

thisDir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(thisDir, '../../../tests'))
import htk_test_utilities as utilities  # noqa
from htk_test_utilities import getTestFilePath, girderClient  # noqa

# # for protyping
# from tests.htk_test_utilities import _connect_to_existing_local_dsa
# girderClient = _connect_to_existing_local_dsa()


class Cfg:
    def __init__(self):
        self.gc = None
        self.folderid = None


cfg = Cfg()

# pytest runs tests in the order they appear in the module
@pytest.mark.usefixtures('girderClient')  # noqa
def test_prep(girderClient):  # noqa

    cfg.gc = girderClient

    # get original item
    iteminfo = cfg.gc.get('/item', parameters={
        'text': "TCGA-A2-A0YE-01Z-00-DX1"})[0]

    # create the folder to "back up"
    folderinfo = cfg.gc.post(
        '/folder', data={
            'parentId': iteminfo['folderId'],
            'name': 'test'
        })
    cfg.folderid = folderinfo['_id']

    # create subfolder to test recursion
    subf = cfg.gc.post(
        '/folder', data={
            'parentId': cfg.folderid,
            'name': 'test_sub'
        })

    # copy the item multiple times to create dummy database
    for i in range(2):
        for fid in (cfg.folderid, subf['_id']):
            _ = cfg.gc.post(
                "/item/%s/copy" % iteminfo['_id'], data={
                    'name': 'test_dbsqlite-%d' % i,
                    'copyAnnotations': True,
                    'folderId': fid,
                })


class TestDatabaseParser:
    """Test girder database parser."""

    def test_dump_annotations_locally_1(self):
        """Test dump annotations locally."""

        savepath = tempfile.mkdtemp()

        # recursively save annotations -- JSONs + sqlite for folders/items
        dump_annotations_locally(
            cfg.gc, folderid=cfg.folderid, local=savepath,
            save_json=True, save_sqlite=True)

        assert set(os.listdir(savepath)) == {
            'test.json', 'test.sqlite',
            'test_dbsqlite-0.json',
            'test_dbsqlite-0_annotations.json',
            'test_dbsqlite-1.json',
            'test_dbsqlite-1_annotations.json',
            'test_sub',
        }

        sql_engine = db.create_engine('sqlite:///%s/test.sqlite' % savepath)
        dbcon = sql_engine.connect()

        result = pd.read_sql_query(
            """SELECT count(*) FROM 'folders';""", dbcon)
        assert int(result.loc[0, :]) == 2

        result = pd.read_sql_query(
            """SELECT count(*) FROM 'items';""", dbcon)
        assert int(result.loc[0, :]) == 4

        # cleanup
        shutil.rmtree(savepath)

    def test_dump_annotations_locally_2(self):
        """Test dump annotations locally."""

        savepath = tempfile.mkdtemp()

        # recursively save annotations -- parse to csv + sqlite
        dump_annotations_locally(
            cfg.gc, folderid=cfg.folderid, local=savepath,
            save_json=False, save_sqlite=True,
            callback=parse_annotations_to_local_tables,
            callback_kwargs={
                'save_csv': True,
                'save_sqlite': True,
            }
        )

        assert set(os.listdir(savepath)), {
            'test.sqlite',
            'test_dbsqlite-0_docs.csv',
            'test_dbsqlite-0_elements.csv',
            'test_dbsqlite-1_docs.csv',
            'test_dbsqlite-1_elements.csv',
            'test_sub'}

        files = [j for j in os.listdir(os.path.join(savepath, 'test_sub'))]
        assert len([j for j in files if j.endswith('_docs.csv')]) == 2
        assert len([j for j in files if j.endswith('_elements.csv')]) == 2

        sql_engine = db.create_engine('sqlite:///%s/test.sqlite' % savepath)
        dbcon = sql_engine.connect()

        result = pd.read_sql_query(
            """SELECT * FROM 'annotation_docs';""", dbcon)
        assert result.shape == (32, 13)
        assert set(result.columns) == {
            '_modelType', '_version', 'annotation_girder_id',
            'created', 'creatorId', 'element_count',
            'element_details', 'groups', 'itemId', 'item_name',
            'public', 'updated', 'updatedId'}

        result = pd.read_sql_query(
            """SELECT * FROM 'annotation_elements';""", dbcon)
        assert result.shape == (304, 13)
        assert set(result.columns) == {
            'annotation_girder_id', 'bbox_area', 'color',
            'coords_x', 'coords_y', 'element_girder_id', 'group', 'label',
            'type', 'xmax', 'xmin', 'ymax', 'ymin'}

        # cleanup
        shutil.rmtree(savepath)
