"""
Created on Sun Aug 11 22:50:03 2019.
@author: tageldim
"""
import copy
import os
import sys

import pytest

from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    get_bboxes_from_slide_annotations, get_image_from_htk_response,
    get_scale_factor_and_appendStr, parse_slide_annotations_into_tables,
    scale_slide_annotations)

thisDir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(thisDir, '../../../'))
from tests.htk_test_utilities import girderClient  # noqa


class Cfg:
    def __init__(self):
        self.gc = None
        self.iteminfo = None
        self.annotations = None


cfg = Cfg()


class TestAnnotAndMaskUtils:
    """Test utilities for annotations and masks."""

    # pytest runs tests in the order they appear in the module
    @pytest.mark.usefixtures('girderClient')  # noqa
    def test_prep(self, girderClient):  # noqa
        cfg.gc = girderClient
        original_iteminfo = cfg.gc.get('/item', parameters={'text': 'TCGA-A2-A0YE-01Z-00-DX1'})[0]

        cfg.folder = cfg.gc.post(
            '/folder', data={
                'parentId': original_iteminfo['folderId'],
                'name': 'test-annot-and-mask'
            })

        # copy the item
        cfg.iteminfo = cfg.gc.post(
            '/item/%s/copy' % original_iteminfo['_id'], data={
                'name': 'TCGA-A2-A0YE-01Z.svs',
                'copyAnnotations': True,
                'folderId': cfg.folder['_id'],
            })
        cfg.annotations = cfg.gc.get('/annotation/item/' + cfg.iteminfo['_id'])

    def test_get_image_from_htk_response(self):
        """Test get_image_from_htk_response."""
        getStr = '/item/%s/tiles/region?left=%d&right=%d&top=%d&bottom=%d&encoding=PNG' % (
            cfg.iteminfo['_id'], 59000, 59100, 35000, 35100)
        resp = cfg.gc.get(getStr, jsonResp=False)
        rgb = get_image_from_htk_response(resp)

        assert rgb.shape == (100, 100, 3)

    def test_get_bboxes_from_slide_annotations(self):
        """Test get_bboxes_from_slide_annotations."""

        element_infos = get_bboxes_from_slide_annotations(
            copy.deepcopy(cfg.annotations))

        assert element_infos.shape == (76, 9)
        assert set(element_infos.columns) == {
            'annidx', 'elementidx', 'type', 'group',
            'xmin', 'xmax', 'ymin', 'ymax', 'bbox_area'}

    def test_parse_slide_annotations_into_tables(self):
        """Test parse_slide_annotations_into_tables."""
        annotation_infos, element_infos = parse_slide_annotations_into_tables(
            copy.deepcopy(cfg.annotations))

        assert set(annotation_infos.columns) == {
            'annotation_girder_id', '_modelType', '_version',
            'itemId', 'created', 'creatorId',
            'public', 'updated', 'updatedId',
            'groups', 'element_count', 'element_details',
        }
        assert set(element_infos.columns) == {
            'annidx', 'annotation_girder_id',
            'elementidx', 'element_girder_id',
            'type', 'group', 'label', 'color',
            'xmin', 'xmax', 'ymin', 'ymax', 'bbox_area',
            'coords_x', 'coords_y'
        }
        assert set(element_infos.loc[:, 'type']) == {'polyline', 'rectangle'}

    def test_scale_slide_annotations(self):
        """Test scale_slide_annotations."""
        for sf in (0.5, 1.0):
            modified = scale_slide_annotations(
                copy.deepcopy(cfg.annotations), sf=sf)
            assert modified[0]['annotation']['elements'][0]['center'] == [
                int(sf * j) for j in
                cfg.annotations[0]['annotation']['elements'][0]['center']]

    def test_get_scale_factor_and_appendStr(self):
        """Test get_scale_factor_and_appendStr."""
        in_out = [
            [(0.2, None), (1.2525, '&mm_x=0.00020000&mm_y=0.00020000')],
            [(None, 10.), (0.25, '&magnification=10.00000000')],
            [(None, None), (1.0, '')],
        ]
        for (MPP, MAG), (sftrue, apstr) in in_out:
            sf, appendStr = get_scale_factor_and_appendStr(
                gc=cfg.gc, slide_id=cfg.iteminfo['_id'], MPP=MPP, MAG=MAG)
            assert sf == sftrue
            assert appendStr == apstr
