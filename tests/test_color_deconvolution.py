import numpy as np
import skimage.io

from histomicstk.preprocessing import color_deconvolution as htk_dcv
from histomicstk.preprocessing.color_deconvolution.separate_stains_macenko_pca import \
    argpercentile as htk_ap

from .datastore import datastore


class TestMacenko:

    def test_macenko(self):
        im_path = datastore.fetch('Easy1.png')
        im = skimage.io.imread(im_path)[..., :3]

        w = htk_dcv.rgb_separate_stains_macenko_pca(im, 255)

        w_expected = [[0.089411, 0.558021, -0.130574],
                      [0.837138, 0.729935, 0.546981],
                      [0.539635, 0.394725, -0.826899]]

        np.testing.assert_allclose(w, w_expected, atol=1e-6)

    def test_argpercentile(self):
        arr = np.array([])
        with np.testing.assert_raises(IndexError):
            htk_ap(arr, 0.5)

        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        w = htk_ap(arr, 0.5)
        w_expected = 5
        np.testing.assert_equal(w, w_expected)

        w = htk_ap(arr, 0.1)
        w_expected = 1
        np.testing.assert_equal(w, w_expected)

        w = htk_ap(arr, 0.0)
        w_expected = 0
        np.testing.assert_equal(w, w_expected)

        w = htk_ap(arr, 0.99)
        w_expected = 9
        np.testing.assert_equal(w, w_expected)


class TestColorDeconvolution:

    def test_roundtrip(self):
        im_path = datastore.fetch('Easy1.png')
        im = skimage.io.imread(im_path)[..., :3]

        w = np.array([[0.650, 0.072, 0],
                      [0.704, 0.990, 0],
                      [0.286, 0.105, 0]])

        conv_result = htk_dcv.color_deconvolution(im, w, 255)

        im_reconv = htk_dcv.color_convolution(conv_result.StainsFloat,
                                              conv_result.Wc, 255)

        np.testing.assert_allclose(im, im_reconv, atol=1)

    def test_short_array(self):
        im_path = datastore.fetch('Easy1.png')
        im = skimage.io.imread(im_path)[..., :3]

        w = [[0.650, 0.072],
             [0.704, 0.990],
             [0.286, 0.105]]

        conv_result = htk_dcv.color_deconvolution(im, w, 255)

        im_reconv = htk_dcv.color_convolution(conv_result.StainsFloat,
                                              conv_result.Wc, 255)
        np.testing.assert_allclose(im, im_reconv, atol=1)
