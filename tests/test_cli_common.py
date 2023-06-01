import collections
import json
import os
from argparse import Namespace

import large_image
import numpy as np
import skimage.io

import histomicstk.preprocessing.color_deconvolution as htk_cdeconv
import histomicstk.preprocessing.color_normalization as htk_cnorm
import histomicstk.segmentation.nuclear as htk_nuclear
import histomicstk.utils as htk_utils
from histomicstk.cli import utils as cli_utils

from .datastore import datastore

GENERATE_GROUNDTRUTH = bool(os.environ.get('GENERATE_GROUNDTRUTH'))


class TestCliCommon:

    def test_get_stain_matrix(self):

        args = Namespace(
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

        ts = large_image.getTileSource(os.path.join(
            datastore.fetch('Easy1.png')))

        result = cli_utils.get_region_dict([-1, -1, -1, -1], 2000, ts)
        expected = {}
        assert result == expected, 'Expected {}, got {}'.format(expected,
                                                                result)

        result = cli_utils.get_region_dict([100, 110, 250, 240], 500, ts)
        expected = dict(region=dict(left=100, top=110, width=250, height=240))
        assert result == expected, 'Expected {}, got {}'.format(expected,
                                                                result)

    def test_segment_wsi_foreground_at_low_res(self):
        np.random.seed(0)

        wsi_path = os.path.join(datastore.fetch(
            'TCGA-06-0129-01Z-00-DX3.bae772ea-dd36-47ec-8185-761989be3cc8.svs'  # noqa
        ))

        ts = large_image.getTileSource(wsi_path)

        im_fgnd_mask_lres, fgnd_seg_scale = \
            cli_utils.segment_wsi_foreground_at_low_res(ts)

        np.testing.assert_equal(fgnd_seg_scale['magnification'], 2.5)

        fgnd_mask_gtruth_file = os.path.join(datastore.fetch(
            'TCGA-06-0129-01Z-00-DX3_fgnd_mask_lres.png'))

        im_fgnd_mask_lres_gtruth = skimage.io.imread(
            fgnd_mask_gtruth_file) > 0

        if GENERATE_GROUNDTRUTH:
            import PIL.Image

            PIL.Image.fromarray(im_fgnd_mask_lres).save(
                '/tmp/TCGA-06-0129-01Z-00-DX3_fgnd_mask_lres.png')

        np.testing.assert_array_equal(im_fgnd_mask_lres > 0,
                                      im_fgnd_mask_lres_gtruth)

    def test_create_tile_nuclei_annotations(self):

        wsi_path = os.path.join(datastore.fetch(
            'TCGA-06-0129-01Z-00-DX3.bae772ea-dd36-47ec-8185-761989be3cc8.svs'  # noqa
        ))

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

            'foreground_threshold': 60,
            'min_radius': 6,
            'max_radius': 12,
            'min_nucleus_area': 25,
            'local_max_search_radius': 8,

            # In Python 3 unittesting, the scheduler fails if it uses processes
            'scheduler': 'multithreading',  # None,
            'num_workers': -1,
            'num_threads_per_worker': 1,
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
               'top': ts_metadata['sizeY'] * 3 / 4,
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
            it_kwargs
        )

        num_fgnd_tiles = np.count_nonzero(
            tile_fgnd_frac_list >= args.min_fgnd_frac)

        np.testing.assert_equal(num_fgnd_tiles, 2)

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

            im_nuclei_stain = im_stains[:, :, 0].astype(float)

            # segment nuclei
            im_nuclei_seg_mask = htk_nuclear.detect_nuclei_kofahi(
                im_nuclei_stain,
                im_nuclei_stain < args.foreground_threshold,
                args.min_radius,
                args.max_radius,
                args.min_nucleus_area,
                args.local_max_search_radius
            )

            # generate nuclei annotations as bboxes
            cur_bbox_annot_list = cli_utils.create_tile_nuclei_annotations(
                im_nuclei_seg_mask, tile_info, 'bbox')

            nuclei_bbox_annot_list.extend(cur_bbox_annot_list)

            # generate nuclei annotations as boundaries
            cur_bndry_annot_list = cli_utils.create_tile_nuclei_annotations(
                im_nuclei_seg_mask, tile_info, 'boundary')

            nuclei_bndry_annot_list.extend(cur_bndry_annot_list)

        if GENERATE_GROUNDTRUTH:
            open('/tmp/TCGA-06-0129-01Z-00-DX3_roi_nuclei_bbox.anot', 'w').write(
                json.dumps({'elements': nuclei_bbox_annot_list}))
            open('/tmp/TCGA-06-0129-01Z-00-DX3_roi_nuclei_boundary.anot', 'w').write(
                json.dumps({'elements': nuclei_bndry_annot_list}))

        # compare nuclei bbox annotations with gtruth
        nuclei_bbox_annot_gtruth_file = os.path.join(datastore.fetch(
            'TCGA-06-0129-01Z-00-DX3_roi_nuclei_bbox.anot'  # noqa
        ))

        with open(nuclei_bbox_annot_gtruth_file) as fbbox_annot:
            nuclei_bbox_annot_list_gtruth = json.load(fbbox_annot)['elements']

        # Check that nuclei_bbox_annot_list is nearly equal to
        # nuclei_bbox_annot_list_gtruth
        assert len(nuclei_bbox_annot_list) == len(nuclei_bbox_annot_list_gtruth)
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
        nuclei_bndry_annot_gtruth_file = os.path.join(datastore.fetch(
            'TCGA-06-0129-01Z-00-DX3_roi_nuclei_boundary.anot'
        ))

        with open(nuclei_bndry_annot_gtruth_file) as fbndry_annot:
            nuclei_bndry_annot_list_gtruth = json.load(
                fbndry_annot)['elements']

        assert len(nuclei_bndry_annot_list) == len(nuclei_bndry_annot_list_gtruth)

        for pos in range(len(nuclei_bndry_annot_list)):

            np.testing.assert_array_almost_equal(
                nuclei_bndry_annot_list[pos]['points'],
                nuclei_bndry_annot_list_gtruth[pos]['points'], 0)

    def test_splitArgs(self):
        args = Namespace(
            a=1,
            b_a=2,
            b_b=3,
            b_c_a=4,
            c_a=5,
        )
        split = cli_utils.splitArgs(args)
        assert split == Namespace(
            a=1,
            b=Namespace(
                a=2,
                b=3,
                c_a=4,
            ),
            c=Namespace(
                a=5,
            ),
        )
