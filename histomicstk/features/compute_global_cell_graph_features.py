from collections import namedtuple

import numpy as np
from numpy import linalg

PopStats = namedtuple('PopStats', ['mean', 'stddev', 'min_max_ratio', 'disorder'])
PolyProps = namedtuple('PolyProps', ['area', 'peri', 'max_dist'])
TriProps = namedtuple('TriProps', ['sides', 'area'])
DensityProps = namedtuple('DensityProps', ['neighbors_in_distance',
                                           'distance_for_neighbors'])
Props = namedtuple('Props', ['voronoi', 'delaunay', 'mst_branches', 'density'])


def compute_global_cell_graph_features(
        centroids,
        neighbor_distances=None,
        neighbor_counts=(3, 5, 7),
):
    r"""Compute global (i.e., not per-nucleus) features of the nuclei with
    the given centroids based on the partitioning of the space into
    Voronoi cells and on the induced graph structure.

    Parameters
    ----------
    centroids : array_like
        Nx2 numpy array of nuclear centroids
    neighbor_distances : array_like
        Radii to count neighbors in
    neighbor_counts : sequence
        Sequence of numbers of neighbors, each of which is used to
        compute statistics relating to the distance required to reach
        that many neighbors.

    Returns
    -------
    props : pandas.DataFrame
        A single-row DataFrame with the following columns:

        - voronoi\_...: Voronoi diagram features

          - area\_...: Polygon area features
          - peri\_...: Polygon perimeter features
          - max_dist\_...: Maximum distance in polygon features

        - delaunay\_...: Delaunay triangulation features

          - sides\_...: Triangle side length features
          - area\_...: Triangle area features

        - mst_branches\_...: Minimum spanning tree branch features
        - density\_...: Density features

          - neighbors_in_distance\_...

            - 0, 1, ..., len(neighbor_distances) - 1: Neighbor count
              within given radius features.

          - distance_for_neighbors\_...

            - 0, 1, ..., len(neighbor_counts) - 1: Minimum distance to
              enclose count neighbors features

        The "..."s are meant to signify that what precedes is the
        start of a column name.  At the end of each column name is one
        of 'mean', 'stddev', 'min_max_ratio', and 'disorder'.
        'min_max_ratio' is the minimum-to-maximum ratio, and disorder
        is stddev / (mean + stddev).

    Note
    ----
    The indices for the density features are with respect to the
    *sorted* values of the corresponding argument sequence.

    References
    ----------
    .. [#] Doyle, S., Agner, S., Madabhushi, A., Feldman, M., & Tomaszewski, J.
       (2008, May).  Automated grading of breast cancer histopathology using
       spectral clustering with textural and architectural image features.
       In Biomedical Imaging: From Nano to Macro, 2008.  ISBI 2008.
       5th IEEE International Symposium on (pp. 496-499).  IEEE.

    """
    if neighbor_distances is None:
        neighbor_distances = 10. * np.arange(1, 6)
    return _flatten_to_dataframe(_compute_global_cell_graph_features(
        centroids,
        neighbor_distances,
        neighbor_counts,
    ))


def _compute_global_cell_graph_features(
        centroids,
        neighbor_distances,
        neighbor_counts,
):
    """Internal support for compute_global_cell_graph_features that
    returns its result in a nested nametuple structure instead of a
    pandas DataFrame.

    """
    from scipy import sparse
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial import Voronoi
    from scipy.spatial import cKDTree as KDTree
    from scipy.spatial.distance import pdist

    vor = Voronoi(centroids)
    centroids = vor.points
    vertices = vor.vertices

    regions = [r for r in vor.regions if r and -1 not in r]
    areas = np.stack(_poly_area(vertices[r]) for r in regions)
    peris = np.stack(_poly_peri(vertices[r]) for r in regions)
    max_dists = np.stack(pdist(vertices[r]).max() for r in regions)
    poly_props = PolyProps._make(map(_pop_stats, (areas, peris, max_dists)))

    # Assume that each Voronoi vertex is on exactly three ridges.
    ridge_points = vor.ridge_points
    # This isn't exactly the collection of sides, since if they should
    # be counted per-triangle then we weight border ridges wrong
    # relative to ridges that are part of two triangles.
    ridge_lengths = _dist(*np.swapaxes(centroids[ridge_points], 0, 1))
    sides = ridge_lengths
    # Point indices of each triangle
    tris = [[] for _ in vertices]
    for p, ri in enumerate(vor.point_region):
        for vi in vor.regions[ri]:
            if vi != -1:
                tris[vi].append(p)
    # This will only fail in particular symmetrical cases where a
    # Voronoi vertex is associated with more than three centroids.
    # Since this should be unlikely in practice, we don't handle it
    # beyond throwing an AssertionError
    assert all(len(t) == 3 for t in tris)
    tris = np.asarray(tris)
    areas = np.stack(_poly_area(centroids[t]) for t in tris)
    tri_props = TriProps._make(map(_pop_stats, (sides, areas)))

    graph = sparse.coo_matrix((ridge_lengths, np.sort(ridge_points).T),
                              (len(centroids), len(centroids)))
    mst = minimum_spanning_tree(graph)
    # Without looking into exactly how minimum_spanning_tree
    # constructs its output, elimate any explicit zeros to be on the
    # safe side.
    mst_branches = _pop_stats(mst.data[mst.data != 0])

    tree = KDTree(centroids)
    neigbors_in_distance = {
        # Yes, we just throw away the actual points
        r: _pop_stats(np.stack(map(len, tree.query_ball_tree(tree, r))) - 1)
        for r in neighbor_distances
    }
    distance_for_neighbors = dict(zip(
        neighbor_counts,
        map(_pop_stats, tree.query(centroids, [c + 1 for c in neighbor_counts])[0].T),
    ))
    density_props = DensityProps(neigbors_in_distance, distance_for_neighbors)

    return Props(poly_props, tri_props, mst_branches, density_props)


def _poly_area(vertices):
    return abs(_poly_signed_area(vertices))


def _poly_signed_area(vertices):
    return .5 * linalg.det(
        np.stack((vertices, np.roll(vertices, -1, axis=-2)), -1)
    ).sum(-1)


def _poly_peri(vertices):
    return _dist(vertices, np.roll(vertices, -1, axis=-2)).sum(-1)


def _dist(x, y):
    """Compute the distance between two sets of points.  Has signature
    (i),(i)->().

    """
    return (np.subtract(x, y) ** 2).sum(-1) ** .5


def _pop_stats(pop):
    # Filter out outliers (here defined as points more than three
    # standard deviations away from the mean)
    while True:
        mean = pop.mean()
        stddev = pop.std()
        mask = abs(pop - mean) <= 3 * stddev
        if mask.all():
            break
        pop = pop[mask]
    minmaxr = pop.min() / pop.max()
    disorder = stddev / (mean + stddev)
    return PopStats(mean, stddev, minmaxr, disorder)


def _flatten_to_dataframe(nt):
    """Flatten the result of _compute_global_cell_graph_features to the
    DataFrame returned by compute_global_cell_graph_features.

    """
    from pandas import DataFrame

    return DataFrame(_flatten_to_dict(nt), index=[0])


def _flatten_to_dict(nt, prefix=''):
    result = {}
    assert isinstance(nt, (tuple, dict))
    if isinstance(nt, tuple):
        d = nt._asdict()
    else:  # nt is a dict
        # We only have numeric keys, and they may be non-nice floats,
        # so just number them by sort order instead of doing something
        # else for names
        d = {str(i): kv[1] for i, kv in enumerate(sorted(nt.items()))}
    for k, v in d.items():
        if not isinstance(v, (tuple, dict)):
            # Terminate
            result[prefix + k] = v
        else:
            result.update(_flatten_to_dict(v, prefix + k + '_'))
    return result
