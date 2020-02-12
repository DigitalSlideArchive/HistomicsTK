# -*- coding: utf-8 -*-
import unittest
import os
import tempfile
import shutil
import pandas as pd
import sqlalchemy as db

from histomicstk.utils.girder_convenience_utils import connect_to_api
from histomicstk.annotations_and_masks.annotation_database_parser import (
    dump_annotations_locally, parse_annotations_to_local_tables)

# %%===========================================================================
# Constants & prep work

gc = connect_to_api(
    apiurl='http://candygram.neurology.emory.edu:8080/api/v1/',
    apikey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')

SAMPLE_FOLDER_ID = "5e24c20dddda5f8398695671"

# %%===========================================================================


class DatabaseParserTest(unittest.TestCase):
    """Test girder database parser."""

    def test_dump_annotations_locally_1(self):
        """Test dump annotations locally."""

        savepath = tempfile.mkdtemp()

        # recursively save annotations -- JSONs + sqlite for folders/items
        dump_annotations_locally(
            gc, folderid=SAMPLE_FOLDER_ID, local=savepath,
            save_json=True, save_sqlite=True)

        self.assertSetEqual(set(os.listdir(savepath)), {
            'Participant_1', 'Participant_2',
            'Concordance.sqlite', 'Concordance.json'}
        )

        sql_engine = db.create_engine(
            'sqlite:///%s/Concordance.sqlite' % savepath)
        dbcon = sql_engine.connect()

        result = pd.read_sql_query(
            """SELECT count(*) FROM 'folders';""", dbcon)
        self.assertEqual(int(result.loc[0, :]), 3)

        result = pd.read_sql_query(
            """SELECT count(*) FROM 'items';""", dbcon)
        self.assertEqual(int(result.loc[0, :]), 10)

        # cleanup
        shutil.rmtree(savepath)

    def test_dump_annotations_locally_2(self):
        """Test dump annotations locally."""

        savepath = tempfile.mkdtemp()

        # recursively save annotations -- parse to csv + sqlite
        dump_annotations_locally(
            gc, folderid=SAMPLE_FOLDER_ID, local=savepath,
            save_json=False, save_sqlite=True,
            callback=parse_annotations_to_local_tables,
            callback_kwargs={
                'save_csv': True,
                'save_sqlite': True,
            }
        )

        self.assertSetEqual(set(os.listdir(savepath)), {
            'Participant_1', 'Participant_2', 'Concordance.sqlite'})

        files = [j for j in os.listdir(os.path.join(savepath, 'Participant_1'))]
        self.assertEqual(
            len([j for j in files if j.endswith('_docs.csv')]), 5)
        self.assertEqual(
            len([j for j in files if j.endswith('_elements.csv')]), 5)

        sql_engine = db.create_engine(
            'sqlite:///%s/Concordance.sqlite' % savepath)
        dbcon = sql_engine.connect()

        result = pd.read_sql_query(
            """SELECT * FROM 'annotation_docs';""", dbcon)
        self.assertTupleEqual(result.shape, (81, 13))
        self.assertSetEqual(set(result.columns), {
            '_modelType', '_version', 'annotation_girder_id',
            'created', 'creatorId', 'element_count',
            'element_details', 'groups', 'itemId', 'item_name',
            'public', 'updated', 'updatedId'})

        result = pd.read_sql_query(
            """SELECT * FROM 'annotation_elements';""", dbcon)
        self.assertTupleEqual(result.shape, (230, 13))
        self.assertSetEqual(set(result.columns), {
            'annotation_girder_id', 'bbox_area', 'color',
            'coords_x', 'coords_y', 'element_girder_id', 'group', 'label',
            'type', 'xmax', 'xmin', 'ymax', 'ymin'})

        # cleanup
        shutil.rmtree(savepath)

# %%===========================================================================


if __name__ == '__main__':
    unittest.main()
