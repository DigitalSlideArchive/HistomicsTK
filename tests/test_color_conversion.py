import numpy as np

from histomicstk.preprocessing.color_conversion import (lab_to_rgb, od_to_rgb,
                                                        rgb_to_lab, rgb_to_od)


class TestColorConversion:

    def test_rgb_to_od(self):

        np.testing.assert_array_almost_equal(
            np.round(rgb_to_od(np.zeros((3, 3, 3)) + 117.0), 4),
            np.zeros((3, 3, 3)) + 35.6158
        )

        # check corner cases
        np.testing.assert_array_almost_equal(
            rgb_to_od(np.zeros((3, 3, 3)) + 255.0),
            np.zeros((3, 3, 3))
        )

        np.testing.assert_array_almost_equal(
            rgb_to_od(np.zeros((3, 3, 3))),
            np.zeros((3, 3, 3)) + 255.0
        )

    def test_od_to_rgb(self):

        np.testing.assert_array_almost_equal(
            od_to_rgb(np.zeros((3, 3, 3)) + 35.6158),
            np.zeros((3, 3, 3)) + 116.99987889
        )

        # check corner cases
        np.testing.assert_array_almost_equal(
            od_to_rgb(np.zeros((3, 3, 3))),
            np.zeros((3, 3, 3)) + 255.0
        )

        np.testing.assert_array_almost_equal(
            od_to_rgb(np.zeros((3, 3, 3)) + 255.0),
            np.zeros((3, 3, 3))
        )

    def test_rgb_to_od_to_rgb(self):

        np.random.seed(1)

        im_rand = np.random.randint(0, 255, (10, 10, 3))

        np.testing.assert_array_almost_equal(
            od_to_rgb(rgb_to_od(im_rand)),
            im_rand
        )

    def test_rgb_to_lab_to_rgb(self):

        np.random.seed(1)

        im_rand = np.random.randint(0, 255, (10, 10, 3))

        np.testing.assert_array_almost_equal(
            np.round(lab_to_rgb(rgb_to_lab(im_rand))),
            im_rand
        )
