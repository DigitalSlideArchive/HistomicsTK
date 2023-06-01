import numpy as np
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from histomicstk.features import compute_global_cell_graph_features as cgcgf


class TestGlobalCellGraphFeatures:
    def testSimple(self):
        data = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1], [-.5, -.5], [.5, .5]])
        actual = cgcgf(data, neighbor_distances=0.7 * np.arange(1, 6), neighbor_counts=(3, 5))
        expected = DataFrame(dict(
            delaunay_area_disorder=0.261203874964,
            delaunay_area_mean=0.666666666667,
            delaunay_area_min_max_ratio=0.5,
            delaunay_area_stddev=0.235702260396,
            delaunay_sides_disorder=0.224914917312,
            delaunay_sides_mean=1.55936204046,
            delaunay_sides_min_max_ratio=0.353553390593,
            delaunay_sides_stddev=0.452497141571,
            density_distance_for_neighbors_0_disorder=0.0959519464365,
            density_distance_for_neighbors_0_mean=1.86037961003,
            density_distance_for_neighbors_0_min_max_ratio=0.790569415042,
            density_distance_for_neighbors_0_stddev=0.197453049082,
            density_distance_for_neighbors_1_disorder=0.113918900724,
            density_distance_for_neighbors_1_mean=2.59272486435,
            density_distance_for_neighbors_1_min_max_ratio=0.75,
            density_distance_for_neighbors_1_stddev=0.333333333333,
            density_neighbors_in_distance_0_disorder=np.nan,
            density_neighbors_in_distance_0_mean=0.0,
            density_neighbors_in_distance_0_min_max_ratio=np.nan,
            density_neighbors_in_distance_0_stddev=0.0,
            density_neighbors_in_distance_1_disorder=0.414213562373,
            density_neighbors_in_distance_1_mean=0.666666666667,
            density_neighbors_in_distance_1_min_max_ratio=0.0,
            density_neighbors_in_distance_1_stddev=0.471404520791,
            density_neighbors_in_distance_2_disorder=0.113918900724,
            density_neighbors_in_distance_2_mean=3.66666666667,
            density_neighbors_in_distance_2_min_max_ratio=0.75,
            density_neighbors_in_distance_2_stddev=0.471404520791,
            density_neighbors_in_distance_3_disorder=0.0981124329991,
            density_neighbors_in_distance_3_mean=4.33333333333,
            density_neighbors_in_distance_3_min_max_ratio=0.8,
            density_neighbors_in_distance_3_stddev=0.471404520791,
            density_neighbors_in_distance_4_disorder=0.0,
            density_neighbors_in_distance_4_mean=5.0,
            density_neighbors_in_distance_4_min_max_ratio=1.0,
            density_neighbors_in_distance_4_stddev=0.0,
            mst_branches_disorder=0.252878170232,
            mst_branches_mean=1.19814095698,
            mst_branches_min_max_ratio=0.4472135955,
            mst_branches_stddev=0.405534520356,
            voronoi_area_disorder=0.0,
            voronoi_area_mean=1.6875,
            voronoi_area_min_max_ratio=1.0,
            voronoi_area_stddev=0.0,
            voronoi_max_dist_disorder=0.0,
            voronoi_max_dist_mean=2.12132034356,
            voronoi_max_dist_min_max_ratio=1.0,
            voronoi_max_dist_stddev=0.0,
            voronoi_peri_disorder=0.0,
            voronoi_peri_mean=5.55368876047,
            voronoi_peri_min_max_ratio=1.0,
            voronoi_peri_stddev=0.0,
        ), index=[0])
        assert_frame_equal(actual, expected, check_like=True)
