from tests import base

import os
import sys
import numpy as np
import skimage.io
import skimage.measure

import collections

import histomicstk.preprocessing.color_normalization as htk_cnorm
import histomicstk.preprocessing.color_deconvolution as htk_cdeconv
import histomicstk.features as htk_features

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), '../server')))
from cli_common import utils as cli_utils  # noqa


# boiler plate to start and stop the server if needed
def setUpModule():

    base.enabledPlugins.append('HistomicsTK')
    base.startServer()


def tearDownModule():

    base.stopServer()


TEST_DATA_DIR = os.path.join(os.environ['GIRDER_TEST_DATA_PREFIX'], 'plugins/HistomicsTK')


class FeatureExtractionTest(base.TestCase):

    def setUp(self):

        # define parameters
        args = {

            'reference_mu_lab': [8.63234435, -0.11501964, 0.03868433],
            'reference_std_lab': [0.57506023, 0.10403329, 0.01364062],

            'min_radius': 12,
            'max_radius': 30,
            'foreground_threshold': 60,
            'min_nucleus_area': 80,
            'local_max_search_radius': 10,
        }

        args = collections.namedtuple('Parameters', args.keys())(**args)

        # read input image
        input_image_file = os.path.join(TEST_DATA_DIR, 'Easy1.png')

        im_input = skimage.io.imread(input_image_file)[:, :, :3]

        # perform color normalization
        im_input_nmzd = htk_cnorm.reinhard(
            im_input, args.reference_mu_lab, args.reference_std_lab)

        # perform color decovolution
        w = htk_cdeconv.rgb_separate_stains_macenko_pca(
            im_input_nmzd, im_input_nmzd.max())

        im_stains = htk_cdeconv.color_deconvolution(im_input_nmzd, w).Stains

        nuclei_channel = htk_cdeconv.find_stain_index(
            htk_cdeconv.stain_color_map['hematoxylin'], w)

        im_nuclei_stain = im_stains[:, :, nuclei_channel].astype(np.float)

        # segment nuclei
        im_nuclei_seg_mask = cli_utils.detect_nuclei_kofahi(
            im_nuclei_stain, args)

        # perform connected component analysis
        nuclei_rprops = skimage.measure.regionprops(im_nuclei_seg_mask)

        self.im_input = im_input
        self.im_input_nmzd = im_input_nmzd
        self.im_nuclei_stain = im_nuclei_stain
        self.im_nuclei_seg_mask = im_nuclei_seg_mask
        self.nuclei_rprops = nuclei_rprops

    def check_fdata_sanity(self, fdata, expected_feature_list):

        self.assertEqual(len(self.nuclei_rprops), fdata.shape[0])
        self.assertEqual(len(expected_feature_list), len(fdata.columns))

        for col in fdata.columns:
            self.assertEqual(col in expected_feature_list, True)

    def test_intensity_features(self):

        from histomicstk.features.compute_intensity_features import \
            feature_list

        fdata = htk_features.compute_intensity_features(
            self.im_nuclei_seg_mask, self.im_nuclei_stain)

        self.check_fdata_sanity(fdata, feature_list)

    def test_haralick_features(self):

        from histomicstk.features.compute_haralick_features import \
            feature_list

        fdata = htk_features.compute_haralick_features(
            self.im_nuclei_seg_mask, self.im_nuclei_stain)

        self.check_fdata_sanity(fdata, feature_list)

    def test_gradient_features(self):

        from histomicstk.features.compute_gradient_features import \
            feature_list

        fdata = htk_features.compute_gradient_features(
            self.im_nuclei_seg_mask, self.im_nuclei_stain)

        self.check_fdata_sanity(fdata, feature_list)

    def test_morphometry_features(self):

        from histomicstk.features.compute_morphometry_features import \
            feature_list

        fdata = htk_features.compute_morphometry_features(
            self.im_nuclei_seg_mask)

        self.check_fdata_sanity(fdata, feature_list)

    def test_fsd_features(self):

        Fs = 6
        feature_list = ['Shape.FSD' + str(i+1) for i in range(Fs)]

        fdata = htk_features.compute_fsd_features(
            self.im_nuclei_seg_mask, self.im_nuclei_stain, Fs=Fs)

        self.check_fdata_sanity(fdata, feature_list)
