import os

import numpy as np
import skimage.io

import histomicstk.cli.NucleiDetection.NucleiDetection as nucl_det
import histomicstk.preprocessing.color_conversion as htk_cvt
import histomicstk.preprocessing.color_deconvolution as htk_cdeconv
import histomicstk.preprocessing.color_normalization as htk_cnorm
import histomicstk.segmentation as htk_seg

from .datastore import datastore


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

    def test_segment_nuclei_with_image_inversion(self):

        # provide input path
        input_grey_img_path = './tests/annotations_and_masks/img/simple.tiff'
        print(os.getcwd())

        # set the args
        args = {}
        args["inputImageFile"] = input_grey_img_path
        args["ImageInversionForm"] = "Yes"
        process_whole_image = True

        # set image inversion flag
        invert_image = nucl_det.image_inversion_flag_setter(args)

        # read the image
        ts, is_wsi = nucl_det.read_input_image(args, process_whole_image)

        # compute tissue foreground mask
        im_fgnd_mask_lres, fgnd_seg_scale = nucl_det.process_wsi_as_whole_image(
            ts, invert_image, args)

        # set the it_kwargs
        it_kwargs = {
            'tile_size': {'width': args.analysis_tile_size},
            'scale': {'magnification': args.analysis_mag}
        }

        # process wsi
        tile_fgnd_frac_list = nucl_det.process_wsi(
            it_kwargs,
            args,
            im_fgnd_mask_lres,
            fgnd_seg_scale,
            process_whole_image)

        # color inversion
        src_mu_lab, src_sigma_lab = nucl_det.compute_reinhard_norm(args, invert_image)

        # detect nuclei with dask
        nuclei_list = nucl_det.detect_nuclei_with_dask(
            ts,
            tile_fgnd_frac_list,
            it_kwargs,
            args,
            invert_image,
            is_wsi,
            src_mu_lab,
            src_sigma_lab)

        # check the number of nuclei
        number_of_nuclei = len(nuclei_list)
        np.testing.assert_allclose(number_of_nuclei, 3166, atol=1e-0)
