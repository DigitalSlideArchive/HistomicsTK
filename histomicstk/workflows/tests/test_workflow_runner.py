#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 18:12:48 2019.

@author: mtageld
"""
import pytest
import os
import sys
from histomicstk.workflows.workflow_runner import Slide_iterator

from histomicstk.utils.girder_convenience_utils import \
    update_styles_for_annotations_in_folder
thisDir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(thisDir, '../../../'))

# import tests.htk_test_utilities as utilities  # noqa
from tests.htk_test_utilities import girderClient  # noqa


# # for protyping
# from tests.htk_test_utilities import _connect_to_existing_local_dsa
# girderClient = _connect_to_existing_local_dsa()

global gc, iteminfo, posted_folder


class TestWorkflows(object):
    """Test slide and folder iterator & workflow runner."""

    # pytest runs tests in the order they appear in the module
    @pytest.mark.usefixtures('girderClient')  # noqa
    def test_prep(self, girderClient):  # noqa
        global gc, posted_folder

        gc = girderClient

        # get original item
        original_iteminfo = gc.get('/item', parameters={
            'text': "TCGA-A2-A0YE-01Z-00-DX1"})[0]

        # create a sample folder
        posted_folder = gc.post(
            '/folder', data={
                'parentId': original_iteminfo['folderId'],
                'name': 'test'
            })

        # copy the item a couple of times
        for i in range(2):
            _ = gc.post(
                "/item/%s/copy" % original_iteminfo['_id'], data={
                    'name': 'TCGA-A2-A0YE_copy-%d' % i,
                    'copyAnnotations': True,
                    'folderId': posted_folder['_id'],
                })

    def test_Slide_iterator(self):
        """Test Slide_iterator.run()."""
        slide_iterator = Slide_iterator(
            gc, source_folder_id=posted_folder['_id'])

        assert len(slide_iterator.slide_ids) == 2

        si = slide_iterator.run()
        for i in range(2):
            slide_info = next(si)
            if i == 0:
                assert(all(
                    [k in slide_info.keys() for k in
                     ('name', '_id', 'levels', 'magnification')]))

    def test_runner_using_annotation_style_update(self):
        """Test workflow runner for cellularity detection."""
        update_styles_for_annotations_in_folder(
            gc, folderid=posted_folder['_id'],
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
