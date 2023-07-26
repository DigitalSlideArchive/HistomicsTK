import argparse
import os
import tempfile

import numpy as np
import skimage.io

import histomicstk.cli.NucleiDetection.NucleiDetection as nucl_det
import histomicstk.preprocessing.color_conversion as htk_cvt
import histomicstk.preprocessing.color_deconvolution as htk_cdeconv
import histomicstk.preprocessing.color_normalization as htk_cnorm
import histomicstk.segmentation as htk_seg

from .datastore import datastore

# Enable to generate groundtruth files in the /tmp directory
GENERATE_GROUNDTRUTH = bool(os.environ.get('GENERATE_GROUNDTRUTH'))


class TestNucleiSegmentation:

    def test_segment_nuclei_kofahi(self):

        input_image_file = datastore.fetch('Easy1.png')

        ref_image_file = datastore.fetch('L1.png')

        # read input image
        im_input = skimage.io.imread(input_image_file)[:, :, :3]

        # read reference image
        im_reference = skimage.io.imread(ref_image_file)[:, :, :3]

        # get mean and stddev of reference image in lab space
        mean_ref, std_ref = htk_cvt.lab_mean_std(im_reference)

        # perform color normalization
        im_nmzd = htk_cnorm.reinhard(im_input, mean_ref, std_ref)

        # perform color decovolution
        stain_color_map = {
            'hematoxylin': [0.65, 0.70, 0.29],
            'eosin': [0.07, 0.99, 0.11],
            'dab': [0.27, 0.57, 0.78],
            'null': [0.0, 0.0, 0.0]
        }

        w = htk_cdeconv.rgb_separate_stains_macenko_pca(im_nmzd, im_nmzd.max())

        im_stains = htk_cdeconv.color_deconvolution(im_nmzd, w).Stains

        nuclei_channel = htk_cdeconv.find_stain_index(stain_color_map['hematoxylin'], w)

        im_nuclei_stain = im_stains[:, :, nuclei_channel].astype(float)

        # segment nuclei
        im_nuclei_seg_mask = htk_seg.nuclear.detect_nuclei_kofahi(
            im_nuclei_stain, im_nuclei_stain < 60,
            min_radius=20, max_radius=30,
            min_nucleus_area=80, local_max_search_radius=10
        )

        num_nuclei = len(np.unique(im_nuclei_seg_mask)) - 1

        # check if segmentation mask matches ground truth
        gtruth_mask_file = os.path.join(datastore.fetch(
            'Easy1_nuclei_seg_kofahi.npy'))

        if GENERATE_GROUNDTRUTH:
            gtruth_mask_file = os.path.join(tempfile.gettempdir(), 'Easy1_nuclei_seg_kofahi.npy')
            np.save(gtruth_mask_file, im_nuclei_seg_mask.astype(np.uint8))

        im_gtruth_mask = np.load(gtruth_mask_file)

        num_nuclei_gtruth = len(np.unique(im_gtruth_mask)) - 1

        assert num_nuclei == num_nuclei_gtruth

        np.testing.assert_allclose(im_nuclei_seg_mask, im_gtruth_mask)

        # check no nuclei case
        im_nuclei_seg_mask = htk_seg.nuclear.detect_nuclei_kofahi(
            255 * np.ones_like(im_nuclei_stain), np.ones_like(im_nuclei_stain),
            min_radius=20, max_radius=30,
            min_nucleus_area=80, local_max_search_radius=10
        )

        num_nuclei = len(np.unique(im_nuclei_seg_mask)) - 1

        assert num_nuclei == 0

    def test_segment_nuclei_guassian_voting(self):

        input_image_file = datastore.fetch('Easy1.png')

        ref_image_file = datastore.fetch('L1.png')

        # read input image
        im_input = skimage.io.imread(input_image_file)[:, :, :3]

        # read reference image
        im_reference = skimage.io.imread(ref_image_file)[:, :, :3]

        # get mean and stddev of reference image in lab space
        mean_ref, std_ref = htk_cvt.lab_mean_std(im_reference)

        # perform color normalization
        im_nmzd = htk_cnorm.reinhard(im_input, mean_ref, std_ref)

        # perform color decovolution
        stain_color_map = {
            'hematoxylin': [0.65, 0.70, 0.29],
            'eosin': [0.07, 0.99, 0.11],
            'dab': [0.27, 0.57, 0.78],
            'null': [0.0, 0.0, 0.0]
        }

        w = htk_cdeconv.rgb_separate_stains_macenko_pca(im_nmzd, im_nmzd.max())

        im_stains = htk_cdeconv.color_deconvolution(im_nmzd, w).Stains

        nuclei_channel = htk_cdeconv.find_stain_index(stain_color_map['hematoxylin'], w)

        im_nuclei_stain = im_stains[:, :, nuclei_channel].astype(float)

        # segment nuclei
        nuclei, votes = htk_seg.nuclear.gaussian_voting(im_nuclei_stain)

        assert len(nuclei.X) > 50
        assert len(votes) > 1000

    # Test arguments
    args = argparse.Namespace(inputImageFile='',
                              ImageInversionForm='Yes',
                              analysis_mag=20.0,
                              analysis_roi=[-1.0, -1.0, -1.0, -1.0],
                              analysis_tile_size=1024.0,
                              foreground_threshold=60.0,
                              frame='0',
                              ignore_border_nuclei=False,
                              local_max_search_radius=10.0,
                              max_radius=20.0,
                              min_fgnd_frac=0.25,
                              min_nucleus_area=80.0,
                              min_radius=6.0,
                              nuclei_annotation_format='boundary',
                              num_threads_per_worker=1,
                              num_workers=-1,
                              reference_mu_lab=[8.63234435,
                                                -0.11501964,
                                                0.03868433],
                              reference_std_lab=[0.57506023,
                                                 0.10403329,
                                                 0.01364062],
                              scheduler='',
                              stain_1='hematoxylin',
                              stain_1_vector=[-1.0,
                                              -1.0,
                                              -1.0],
                              stain_2='eosin',
                              stain_2_vector=[-1.0,
                                              -1.0,
                                              -1.0],
                              stain_3='null',
                              stain_3_vector=[-1.0,
                                              -1.0,
                                              -1.0],
                              style=None,
                              tile_overlap_value=0)

    def test_image_inversion_flag_setter(self):
        invert_image, default_img_inversion = nucl_det.image_inversion_flag_setter(self.args)
        np.testing.assert_equal(invert_image, True)
        np.testing.assert_equal(default_img_inversion, False)

    def test_nuclei_detection(self):

        # retrieve image from datastore
        input_image_path = datastore.fetch('tcgaextract_ihergb.tiff')
        self.args.inputImageFile = input_image_path

        # read the image
        ts, is_wsi = nucl_det.read_input_image(self.args, process_whole_image=True)
        it_kwargs = {
            'tile_size': {'width': self.args.analysis_tile_size},
            'scale': {'magnification': self.args.analysis_mag}
        }

        # determine number of nuclei
        tile_fgnd_frac_list = nucl_det.process_wsi(ts, it_kwargs, self.args)
        nuclei_list = nucl_det.detect_nuclei_with_dask(
            ts,
            tile_fgnd_frac_list,
            it_kwargs,
            self.args,
            invert_image=False,
            is_wsi=is_wsi,
            src_mu_lab=None,
            src_sigma_lab=None)
        np.testing.assert_allclose(len(nuclei_list), 10, 1e+2)

    def test_nuclei_detection_image_inverted(self):

        # retrieve image from datastore
        input_image_path = datastore.fetch('tcgaextract_ihergb.tiff')
        self.args.inputImageFile = input_image_path

        # read the image
        ts, is_wsi = nucl_det.read_input_image(self.args, process_whole_image=True)
        it_kwargs = {
            'tile_size': {'width': self.args.analysis_tile_size},
            'scale': {'magnification': self.args.analysis_mag}
        }

        # determine number of nuclei
        tile_fgnd_frac_list = nucl_det.process_wsi(ts, it_kwargs, self.args)
        nuclei_list = nucl_det.detect_nuclei_with_dask(
            ts,
            tile_fgnd_frac_list,
            it_kwargs,
            self.args,
            invert_image=True,
            is_wsi=is_wsi,
            src_mu_lab=None,
            src_sigma_lab=None)
        np.testing.assert_allclose(len(nuclei_list), 7497, 1e+2)

    def test_tile_overlap(self):
        # retrieve image from datastore
        input_image_path = datastore.fetch('tcgaextract_ihergb.tiff')
        self.args.inputImageFile = input_image_path
        self.args.tile_overlap_value = 128

        # read the image
        ts, is_wsi = nucl_det.read_input_image(self.args, process_whole_image=True)
        it_kwargs = {
            'tile_size': {'width': self.args.analysis_tile_size},
            'scale': {'magnification': self.args.analysis_mag},
            'tile_overlap': {'x': self.args.tile_overlap_value, 'y': self.args.tile_overlap_value},
        }
        # determine number of nuclei
        tile_fgnd_frac_list = nucl_det.process_wsi(ts, it_kwargs, self.args)
        nuclei_list = nucl_det.detect_nuclei_with_dask(
            ts,
            tile_fgnd_frac_list,
            it_kwargs,
            self.args,
            invert_image=True,
            is_wsi=is_wsi,
            src_mu_lab=None,
            src_sigma_lab=None)
        np.testing.assert_allclose(len(nuclei_list), 3000, 1e+2)
