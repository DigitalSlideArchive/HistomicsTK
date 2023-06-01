import collections
import os
import sys
import tempfile

import packaging.version
import pandas as pd
import skimage.io
import skimage.measure

import histomicstk.features as htk_features
import histomicstk.preprocessing.color_deconvolution as htk_cdeconv
import histomicstk.preprocessing.color_normalization as htk_cnorm
import histomicstk.segmentation.nuclear as htk_nuclear

thisDir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, thisDir)
import htk_test_utilities as utilities  # noqa

try:
    from .datastore import datastore  # noqa
except ModuleNotFoundError:  # local prototyping
    from datastore import datastore  # noqa


class Cfg:
    def __init__(self):
        self.fdata_nuclei = None
        self.im_nuclei_stain = None
        self.im_nuclei_seg_mask = None
        self.nuclei_rprops = None


cfg = Cfg()
# Enable to generate groundtruth files in the /tmp directory
GENERATE_GROUNDTRUTH = bool(os.environ.get('GENERATE_GROUNDTRUTH'))


def check_fdata_sanity(fdata, expected_feature_list,
                       prefix='', match_feature_count=True):

    assert len(cfg.nuclei_rprops) == fdata.shape[0]

    if len(prefix) > 0:
        fcols = [col for col in fdata.columns if col.startswith(prefix)]
    else:
        fcols = fdata.columns

    if match_feature_count:
        assert len(fcols) == len(expected_feature_list)

    for col in expected_feature_list:
        assert prefix + col in fcols


class TestFeatureExtraction:

    def test_setup(self):

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
        input_image_file = datastore.fetch('Easy1.png')

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

        im_nuclei_stain = im_stains[:, :, nuclei_channel].astype(float)

        cytoplasm_channel = htk_cdeconv.find_stain_index(
            htk_cdeconv.stain_color_map['eosin'], w)

        im_cytoplasm_stain = im_stains[:, :, cytoplasm_channel].astype(
            float)

        # segment nuclei
        im_nuclei_seg_mask = htk_nuclear.detect_nuclei_kofahi(
            im_nuclei_stain,
            im_nuclei_stain < args.foreground_threshold,
            args.min_radius,
            args.max_radius,
            args.min_nucleus_area,
            args.local_max_search_radius
        )

        # perform connected component analysis
        nuclei_rprops = skimage.measure.regionprops(im_nuclei_seg_mask)

        # compute nuclei features
        fdata_nuclei = htk_features.compute_nuclei_features(
            im_nuclei_seg_mask, im_nuclei_stain,
            im_cytoplasm=im_cytoplasm_stain)

        cfg.im_input = im_input
        cfg.im_input_nmzd = im_input_nmzd
        cfg.im_nuclei_stain = im_nuclei_stain
        cfg.im_nuclei_seg_mask = im_nuclei_seg_mask
        cfg.nuclei_rprops = nuclei_rprops
        cfg.fdata_nuclei = fdata_nuclei

    def test_compute_intensity_features(self):

        expected_feature_list = [
            'Intensity.Min',
            'Intensity.Max',
            'Intensity.Mean',
            'Intensity.Median',
            'Intensity.MeanMedianDiff',
            'Intensity.Std',
            'Intensity.IQR',
            'Intensity.MAD',
            'Intensity.Skewness',
            'Intensity.Kurtosis',
            'Intensity.HistEnergy',
            'Intensity.HistEntropy',
        ]

        fdata = htk_features.compute_intensity_features(
            cfg.im_nuclei_seg_mask, cfg.im_nuclei_stain)

        check_fdata_sanity(fdata, expected_feature_list)

        check_fdata_sanity(
            cfg.fdata_nuclei, expected_feature_list,
            prefix='Nucleus.', match_feature_count=False)

        check_fdata_sanity(
            cfg.fdata_nuclei, expected_feature_list,
            prefix='Cytoplasm.', match_feature_count=False)

        if GENERATE_GROUNDTRUTH:
            fdata.to_csv(os.path.join(
                tempfile.gettempdir(), 'Easy1_nuclei_intensity_features.csv'),
                index=False)

        fdata_gtruth = pd.read_csv(
            utilities.getTestFilePath('Easy1_nuclei_intensity_features.csv'),
            index_col=None)

        pd.testing.assert_frame_equal(
            fdata, fdata_gtruth, atol=1e-2)

    def test_compute_haralick_features(self):

        f = [
            'Haralick.ASM',
            'Haralick.Contrast',
            'Haralick.Correlation',
            'Haralick.SumOfSquares',
            'Haralick.IDM',
            'Haralick.SumAverage',
            'Haralick.SumVariance',
            'Haralick.SumEntropy',
            'Haralick.Entropy',
            'Haralick.DifferenceVariance',
            'Haralick.DifferenceEntropy',
            'Haralick.IMC1',
            'Haralick.IMC2',
        ]

        expected_feature_list = []
        for col in f:
            expected_feature_list.append(col + '.Mean')
            expected_feature_list.append(col + '.Range')

        fdata = htk_features.compute_haralick_features(
            cfg.im_nuclei_seg_mask, cfg.im_nuclei_stain)

        check_fdata_sanity(fdata, expected_feature_list)

        check_fdata_sanity(
            cfg.fdata_nuclei, expected_feature_list,
            prefix='Nucleus.', match_feature_count=False)

        check_fdata_sanity(
            cfg.fdata_nuclei, expected_feature_list,
            prefix='Cytoplasm.', match_feature_count=False)

        if GENERATE_GROUNDTRUTH:
            fdata.to_csv(os.path.join(
                tempfile.gettempdir(), 'Easy1_nuclei_haralick_features.csv'),
                index=False)

        fdata_gtruth = pd.read_csv(
            utilities.getTestFilePath('Easy1_nuclei_haralick_features.csv'))

        pd.testing.assert_frame_equal(
            fdata, fdata_gtruth, atol=1e-2)

    def test_compute_gradient_features(self):

        expected_feature_list = [
            'Gradient.Mag.Mean',
            'Gradient.Mag.Std',
            'Gradient.Mag.Skewness',
            'Gradient.Mag.Kurtosis',
            'Gradient.Mag.HistEntropy',
            'Gradient.Mag.HistEnergy',
            'Gradient.Canny.Sum',
            'Gradient.Canny.Mean',
        ]

        fdata = htk_features.compute_gradient_features(
            cfg.im_nuclei_seg_mask, cfg.im_nuclei_stain)

        check_fdata_sanity(fdata, expected_feature_list)

        check_fdata_sanity(
            cfg.fdata_nuclei, expected_feature_list,
            prefix='Nucleus.', match_feature_count=False)

        check_fdata_sanity(
            cfg.fdata_nuclei, expected_feature_list,
            prefix='Cytoplasm.', match_feature_count=False)

        if GENERATE_GROUNDTRUTH:
            fdata.to_csv(os.path.join(
                tempfile.gettempdir(), 'Easy1_nuclei_gradient_features.csv'),
                index=False)

        test_file = 'Easy1_nuclei_gradient_features.csv'
        if (
            packaging.version.parse(skimage.__version__) >= packaging.version.parse('0.19') and
            packaging.version.parse(skimage.__version__) < packaging.version.parse('0.19.3')
        ):
            test_file = 'Easy1_nuclei_gradient_features_v2.csv'
        fdata_gtruth = pd.read_csv(
            utilities.getTestFilePath(test_file),
            index_col=None)

        pd.testing.assert_frame_equal(
            fdata, fdata_gtruth, atol=1e-2)

    def test_compute_morphometry_features(self):

        expected_feature_list = [
            'Orientation.Orientation',
            'Size.Area',
            'Size.ConvexHullArea',
            'Size.MajorAxisLength',
            'Size.MinorAxisLength',
            'Size.Perimeter',
            'Shape.Circularity',
            'Shape.Eccentricity',
            'Shape.EquivalentDiameter',
            'Shape.Extent',
            'Shape.FractalDimension',
            'Shape.MinorMajorAxisRatio',
            'Shape.Solidity',
            'Shape.HuMoments1',
            'Shape.HuMoments2',
            'Shape.HuMoments3',
            'Shape.HuMoments4',
            'Shape.HuMoments5',
            'Shape.HuMoments6',
            'Shape.HuMoments7',
        ]

        fdata = htk_features.compute_morphometry_features(
            cfg.im_nuclei_seg_mask)

        check_fdata_sanity(fdata, expected_feature_list)

        check_fdata_sanity(
            cfg.fdata_nuclei, expected_feature_list, match_feature_count=False)

        if GENERATE_GROUNDTRUTH:
            fdata.to_csv(os.path.join(
                tempfile.gettempdir(),
                'Easy1_nuclei_morphometry_features.csv'),
                index=False)

        test_file = 'Easy1_nuclei_morphometry_features.csv'
        if packaging.version.parse(skimage.__version__) >= packaging.version.parse('0.18'):
            test_file = 'Easy1_nuclei_morphometry_features_v2.csv'
        fdata_gtruth = pd.read_csv(
            utilities.getTestFilePath(test_file),
            index_col=None)
        pd.testing.assert_frame_equal(
            fdata, fdata_gtruth, atol=1e-2)

    def test_compute_fsd_features(self):

        Fs = 6
        expected_feature_list = ['Shape.FSD' + str(i + 1) for i in range(Fs)]

        fdata = htk_features.compute_fsd_features(
            cfg.im_nuclei_seg_mask, Fs=Fs)

        check_fdata_sanity(fdata, expected_feature_list)

        check_fdata_sanity(
            cfg.fdata_nuclei, expected_feature_list, match_feature_count=False)

        if GENERATE_GROUNDTRUTH:
            fdata.to_csv(os.path.join(
                tempfile.gettempdir(), 'Easy1_nuclei_fsd_features.csv'),
                index=False)

        fdata_gtruth = pd.read_csv(
            utilities.getTestFilePath('Easy1_nuclei_fsd_features.csv'))

        pd.testing.assert_frame_equal(
            fdata, fdata_gtruth, atol=1e-2)


if __name__ == '__main__':

    tfe = TestFeatureExtraction()
    tfe.test_setup()
    tfe.test_compute_morphometry_features()
    tfe.test_compute_intensity_features()
