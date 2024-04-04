import numpy as np

from histomicstk.segmentation.label import (delete_border,
                                            trace_object_boundaries)


class TestTraceBoundary:

    def test_trace_boundary(self):

        # test moore neighbor algorithm

        m_neighbor = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                               [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                               [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                               [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                               [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                               [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)

        # refenece neighbors for isbf
        rx_isbf = [1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 8, 9, 9, 8, 7, 7, 7,
                   7, 6, 6, 5, 5, 5, 4, 4, 3, 3, 3, 3, 2, 1]
        ry_isbf = [7, 8, 8, 7, 6, 6, 6, 6, 6, 7, 8, 8, 7, 7, 6, 5, 4,
                   3, 3, 2, 2, 1, 2, 2, 3, 3, 4, 5, 6, 7, 7]

        # reference neighbors for moore
        rx_moore = [1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 8, 9, 9, 8, 7, 7, 7,
                    7, 6, 5, 4, 3, 3, 3, 3, 2, 1]
        ry_moore = [7, 8, 8, 7, 6, 6, 6, 6, 6, 7, 8, 8, 7, 7, 6, 5, 4,
                    3, 2, 1, 2, 3, 4, 5, 6, 7, 7]

        x_isbf, y_isbf = trace_object_boundaries(
            m_neighbor, simplify_colinear_spurs=False)

        np.testing.assert_allclose(rx_isbf, x_isbf[0])
        np.testing.assert_allclose(ry_isbf, y_isbf[0])

        x_moore, y_moore = trace_object_boundaries(
            m_neighbor, 8, simplify_colinear_spurs=False)

        np.testing.assert_allclose(rx_moore, x_moore[0])
        np.testing.assert_allclose(ry_moore, y_moore[0])


class TestDeleteBorderLabel:

    def test_delete_border(self):

        # check for correct deletion in an image that has border objects
        im_label = np.zeros((10, 10), dtype='int')
        im_label[0:3, 4:7] = 1
        im_label[8:10, 4:7] = 2
        im_label[4:7, 0:3] = 3
        im_label[4:7, 8:10] = 4

        im_label_del = delete_border(im_label)

        np.testing.assert_array_equal(im_label_del, np.zeros_like(im_label))

        # check if it returns original mask if there are no border objects
        im_label = np.zeros((10, 10), dtype='int')
        im_label[4:6, 4:6] = 1

        im_label_del = delete_border(im_label)

        np.testing.assert_array_equal(im_label_del, im_label)
