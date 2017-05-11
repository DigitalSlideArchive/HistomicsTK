#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
#  Copyright Kitware Inc.
#
#  Licensed under the Apache License, Version 2.0 ( the "License" );
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

# This is to serve as an example for how to create a server-side test in a
# girder plugin, it is not meant to be useful.

from tests import base

import os
import sys
import json
import collections

import numpy as np
import skimage.io

import large_image

import histomicstk.preprocessing.color_deconvolution as htk_cdeconv
import histomicstk.preprocessing.color_normalization as htk_cnorm
import histomicstk.utils as htk_utils

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), '../server')))
from cli_common import utils as cli_utils  # noqa

TEST_DATA_DIR = os.path.join(os.environ['GIRDER_TEST_DATA_PREFIX'], 'plugins/HistomicsTK')


# boiler plate to start and stop the server
def setUpModule():
    base.enabledPlugins.append('HistomicsTK')
    base.startServer()


def tearDownModule():
    base.stopServer()


class Namespace(object):
    pass


class CliCommonTest(base.TestCase):

    def test_get_stain_matrix(self):

        args = Namespace()
        args.__dict__ = dict(
            stain_1='hematoxylin',
            stain_1_vector=[-1, -1, -1],
            stain_2='custom',
            stain_2_vector=[0.1, 0.2, 0.3],
        )
        expected = np.array([htk_cdeconv.stain_color_map['hematoxylin'],
                             [0.1, 0.2, 0.3]]).T
        np.testing.assert_allclose(cli_utils.get_stain_matrix(args, 2),
                                   expected)

    def test_get_region_dict(self):

        ts = large_image.getTileSource(os.path.join(TEST_DATA_DIR,
                                                    'Easy1.png'))

        result = cli_utils.get_region_dict([-1, -1, -1, -1], 2000, ts)
        expected = {}
        assert result == expected, "Expected {}, got {}".format(expected,
                                                                result)

        result = cli_utils.get_region_dict([100, 110, 250, 240], 500, ts)
        expected = dict(region=dict(left=100, top=110, width=250, height=240))
        assert result == expected, "Expected {}, got {}".format(expected,
                                                                result)

    def test_segment_wsi_foreground_at_low_res(self):

        wsi_path = os.path.join(
            TEST_DATA_DIR,
            'TCGA-02-0010-01Z-00-DX4.07de2e55-a8fe-40ee-9e98-bcb78050b9f7.svs'
        )

        ts = large_image.getTileSource(wsi_path)

        im_fgnd_mask_lres, fgnd_seg_scale = \
            cli_utils.segment_wsi_foreground_at_low_res(ts)

        np.testing.assert_equal(fgnd_seg_scale['magnification'], 1.25)

        fgnd_mask_gtruth_file = os.path.join(
            TEST_DATA_DIR,
            'TCGA-02-0010-01Z-00-DX4_fgnd_mask_lres.png'
        )

        im_fgnd_mask_lres_gtruth = skimage.io.imread(
            fgnd_mask_gtruth_file) > 0

        np.testing.assert_array_equal(im_fgnd_mask_lres > 0,
                                      im_fgnd_mask_lres_gtruth)

    def test_create_tile_nuclei_annotations(self):

        wsi_path = os.path.join(
            TEST_DATA_DIR,
            'TCGA-02-0010-01Z-00-DX4.07de2e55-a8fe-40ee-9e98-bcb78050b9f7.svs'
        )

        # define parameters
        args = {

            'reference_mu_lab': [8.63234435, -0.11501964, 0.03868433],
            'reference_std_lab': [0.57506023, 0.10403329, 0.01364062],

            'stain_1': 'hematoxylin',
            'stain_2': 'eosin',
            'stain_3': 'null',

            'stain_1_vector': [-1, -1, -1],
            'stain_2_vector': [-1, -1, -1],
            'stain_3_vector': [-1, -1, -1],

            'min_fgnd_frac': 0.50,
            'analysis_mag': 20,
            'analysis_tile_size': 1200,

            'min_radius': 12,
            'max_radius': 30,
            'foreground_threshold': 60,
            'min_nucleus_area': 80,
            'local_max_search_radius': 10,

            'scheduler_address': None
        }

        args = collections.namedtuple('Parameters', args.keys())(**args)

        # read WSI
        ts = large_image.getTileSource(wsi_path)

        ts_metadata = ts.getMetadata()

        analysis_tile_size = {
            'width': int(ts_metadata['tileWidth'] * np.floor(
                1.0 * args.analysis_tile_size / ts_metadata['tileWidth'])),
            'height': int(ts_metadata['tileHeight'] * np.floor(
                1.0 * args.analysis_tile_size / ts_metadata['tileHeight']))
        }

        # define ROI
        roi = {'left': ts_metadata['sizeX'] / 2,
               'top': ts_metadata['sizeY'] / 2,
               'width': analysis_tile_size['width'],
               'height': analysis_tile_size['height'],
               'units': 'base_pixels'}

        # define tile iterator parameters
        it_kwargs = {
            'tile_size': {'width': args.analysis_tile_size},
            'scale': {'magnification': args.analysis_mag},
            'region': roi
        }

        # create dask client
        cli_utils.create_dask_client(args)

        # get tile foregreoung at low res
        im_fgnd_mask_lres, fgnd_seg_scale = \
            cli_utils.segment_wsi_foreground_at_low_res(ts)

        # compute tile foreground fraction
        tile_fgnd_frac_list = htk_utils.compute_tile_foreground_fraction(
            wsi_path, im_fgnd_mask_lres, fgnd_seg_scale,
            **it_kwargs
        )

        num_fgnd_tiles = np.count_nonzero(
            tile_fgnd_frac_list >= args.min_fgnd_frac)

        np.testing.assert_equal(num_fgnd_tiles, 4)

        # create nuclei annotations
        nuclei_bbox_annot_list = []
        nuclei_bndry_annot_list = []

        for tile_info in ts.tileIterator(
                format=large_image.tilesource.TILE_FORMAT_NUMPY,
                **it_kwargs):

            im_tile = tile_info['tile'][:, :, :3]

            # perform color normalization
            im_nmzd = htk_cnorm.reinhard(im_tile,
                                         args.reference_mu_lab,
                                         args.reference_std_lab)

            # perform color deconvolution
            w = cli_utils.get_stain_matrix(args)

            im_stains = htk_cdeconv.color_deconvolution(im_nmzd, w).Stains

            im_nuclei_stain = im_stains[:, :, 0].astype(np.float)

            # segment nuclei
            im_nuclei_seg_mask = cli_utils.detect_nuclei_kofahi(
                im_nuclei_stain, args)

            # generate nuclei annotations as bboxes
            cur_bbox_annot_list = cli_utils.create_tile_nuclei_annotations(
                im_nuclei_seg_mask, tile_info, 'bbox')

            nuclei_bbox_annot_list.extend(cur_bbox_annot_list)

            # generate nuclei annotations as boundaries
            cur_bndry_annot_list = cli_utils.create_tile_nuclei_annotations(
                im_nuclei_seg_mask, tile_info, 'boundary')

            nuclei_bndry_annot_list.extend(cur_bndry_annot_list)

        # compare nuclei bbox annotations with gtruth
        nuclei_bbox_annot_gtruth_file = os.path.join(
            TEST_DATA_DIR,
            'TCGA-02-0010-01Z-00-DX4_roi_nuclei_bbox.anot'  # noqa
        )

        with open(nuclei_bbox_annot_gtruth_file, 'r') as fbbox_annot:
            nuclei_bbox_annot_list_gtruth = json.load(fbbox_annot)['elements']

        # Check that nuclei_bbox_annot_list is nearly equal to
        # nuclei_bbox_annot_list_gtruth
        self.assertEqual(len(nuclei_bbox_annot_list),
                         len(nuclei_bbox_annot_list_gtruth))
        for pos in range(len(nuclei_bbox_annot_list)):
            np.testing.assert_array_almost_equal(
                nuclei_bbox_annot_list[pos]['center'],
                nuclei_bbox_annot_list_gtruth[pos]['center'], 0)
            np.testing.assert_almost_equal(
                nuclei_bbox_annot_list[pos]['width'],
                nuclei_bbox_annot_list_gtruth[pos]['width'], 1)
            np.testing.assert_almost_equal(
                nuclei_bbox_annot_list[pos]['height'],
                nuclei_bbox_annot_list_gtruth[pos]['height'], 1)

        # compare nuclei boundary annotations with gtruth
        nuclei_bndry_annot_gtruth_file = os.path.join(
            TEST_DATA_DIR,
            'TCGA-02-0010-01Z-00-DX4_roi_nuclei_boundary.anot'  # noqa
        )

        with open(nuclei_bndry_annot_gtruth_file, 'r') as fbndry_annot:
            nuclei_bndry_annot_list_gtruth = json.load(fbndry_annot)['elements']

        assert nuclei_bndry_annot_list == nuclei_bndry_annot_list_gtruth
