import numpy as np
import skimage.feature

import histomicstk as htk


class TestGLCMMatrixGeneration:

    def test_graycomatrixext(self):

        image = np.array([[0, 0, 1, 1],
                          [0, 0, 1, 1],
                          [0, 2, 2, 2],
                          [2, 2, 3, 3]], dtype=np.uint8)

        # Work with skimage < 1.19 and >= 1.19
        if not hasattr(skimage.feature, 'graycomatrix'):
            skimage.feature.graycomatrix = skimage.feature.greycomatrix

        # test 0, 45, 90, 135 degree offsets
        res_skim = skimage.feature.graycomatrix(
            image, [1],
            [0, np.pi / 4.0, np.pi / 2.0, 3 * np.pi / 4.0],
            levels=4
        )

        res_htk = htk.features.graycomatrixext(
            image, num_levels=4, gray_limits=[0, 3],
            offsets=np.array([[0, 1], [1, 1], [1, 0], [1, -1]])
        )

        np.testing.assert_allclose(np.squeeze(res_htk), np.squeeze(res_skim))

        # test 0, 45, 90, 135 degree offsets - normalized
        res_skim = skimage.feature.graycomatrix(
            image, [1],
            [0, np.pi / 4.0, np.pi / 2.0, 3 * np.pi / 4.0],
            levels=4, normed=True
        )

        res_htk = htk.features.graycomatrixext(
            image, num_levels=4, gray_limits=[0, 3], normed=True,
            offsets=np.array([[0, 1], [1, 1], [1, 0], [1, -1]])
        )

        np.testing.assert_allclose(np.squeeze(res_htk), np.squeeze(res_skim))

        # test 0, 45, 90, 135 degree offsets - symmetric
        res_skim = skimage.feature.graycomatrix(
            image, [1],
            [0, np.pi / 4.0, np.pi / 2.0, 3 * np.pi / 4.0],
            levels=4, symmetric=True
        )

        res_htk = htk.features.graycomatrixext(
            image, num_levels=4, gray_limits=[0, 3], symmetric=True,
            offsets=np.array([[0, 1], [1, 1], [1, 0], [1, -1]])
        )

        np.testing.assert_allclose(np.squeeze(res_htk), np.squeeze(res_skim))
