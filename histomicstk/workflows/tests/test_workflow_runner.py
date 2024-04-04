"""
Created on Mon Sep 30 18:12:48 2019.

@author: mtageld
"""
import os
import sys

import pytest

from histomicstk.utils.girder_convenience_utils import \
    update_styles_for_annotations_in_folder
from histomicstk.workflows.workflow_runner import Slide_iterator

thisDir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(thisDir, '../../../'))
from tests.htk_test_utilities import girderClient  # noqa


class Cfg:
    def __init__(self):
        self.gc = None
        self.posted_folder = None


cfg = Cfg()


class TestWorkflows:
    """Test slide and folder iterator & workflow runner."""

    # pytest runs tests in the order they appear in the module
    @pytest.mark.usefixtures('girderClient')  # noqa
    def test_prep(self, girderClient):  # noqa

        cfg.gc = girderClient

        # get original item
        original_iteminfo = cfg.gc.get('/item', parameters={'text': 'TCGA-A2-A0YE-01Z-00-DX1'})[0]

        # create a sample folder
        cfg.posted_folder = cfg.gc.post(
            '/folder', data={
                'parentId': original_iteminfo['folderId'],
                'name': 'test-workflow',
            })

        # copy the item a couple of times
        for i in range(2):
            _ = cfg.gc.post(
                '/item/%s/copy' % original_iteminfo['_id'], data={
                    'name': 'TCGA-A2-A0YE_copy-%d' % i,
                    'copyAnnotations': True,
                    'folderId': cfg.posted_folder['_id'],
                })

    def test_Slide_iterator(self):
        """Test Slide_iterator.run()."""
        slide_iterator = Slide_iterator(
            cfg.gc, source_folder_id=cfg.posted_folder['_id'])

        assert len(slide_iterator.slide_ids) == 2

        si = slide_iterator.run()
        for i in range(2):
            slide_info = next(si)
            if i == 0:
                assert all(
                    k in slide_info.keys() for k in
                    ('name', '_id', 'levels', 'magnification'))

    def test_runner_using_annotation_style_update(self):
        """Test workflow runner for cellularity detection."""
        update_styles_for_annotations_in_folder(
            cfg.gc, folderid=cfg.posted_folder['_id'],
            workflow_kwargs={
                'changes': {
                    'roi': {
                        'group': 'modified_roi',
                        'lineColor': 'rgb(0,0,255)',
                        'fillColor': 'rgba(0,0,255,0.3)',
                    },
                },
            },
            recursive=True, catch_exceptions=True,
            monitor='test', verbose=0,
        )
        # just make sure it runs without exceptions and yo're good.
