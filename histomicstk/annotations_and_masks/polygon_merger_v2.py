# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 16:01:26 2019.

@author: tageldim
"""

import numpy as np
from pandas import DataFrame  # , concat
from shapely.geometry.polygon import Polygon
from shapely.ops import cascaded_union
from histomicstk.annotations_and_masks.masks_to_annotations_handler import (
    _parse_annot_coords, )
from histomicstk.annotations_and_masks.pyrtree.rtree import RTree, Rect
from histomicstk.utils.general_utils import Base_HTK_Class

# %% =====================================================================


class Polygon_merger_v2(Base_HTK_Class):
    """Methods to merge contiguous polygons from whole-slide image."""

    def __init__(self, contours_df, **kwargs):
        """Init Polygon_merger object.

        Arguments:
        -----------
        contours_df : pandas DataFrame
            The following columns are needed.

            group : str
                annotation group (ground truth label).
            ymin : int
                minimun y coordinate
            ymax : int
                maximum y coordinate
            xmin : int
                minimum x coordinate
            xmax : int
                maximum x coordinate
            coords_x : str
                vertix x coordinates comma-separated values
            coords_y
                vertix y coordinated comma-separated values
        merge_thresh : int
            how close do the polygons need to be (in pixels) to be merged
        verbose : int
            0 - Do not print to screen
            1 - Print only key messages
            2 - Print everything to screen
        monitorPrefix : str
            text to prepend to printed statements

        """
        # see: https://stackoverflow.com/questions/8187082/how-can-you-set-...
        # class-attributes-from-variable-arguments-kwargs-in-python
        default_attr = {
            'verbose': 1,
            'monitorPrefix': "",
            'merge_thresh': 3,
        }
        default_attr.update(kwargs)
        super(Polygon_merger_v2, self).__init__(default_attr=default_attr)

        # This is where contours will be stored
        self.contours_df = contours_df
        self.contours_df.reset_index(inplace=True, drop=True)
        self.new_contours = DataFrame(columns=self.contours_df.columns)

        # prepwork
        self.buffer_size = self.merge_thresh + 3
        self.unique_groups = set(self.contours_df.loc[:, "group"])

    # %% =====================================================================

    def set_contours_slice(self, group):
        """Slice a single group from self.contours_df."""
        self.contours_slice = self.contours_df.loc[
            self.contours_df.loc[:, "group"] == group, :]

    # %% =====================================================================

    def create_rtree(self):
        """Add contour bounding boxes to R-tree."""
        self.rtree = RTree()
        for cidx, cont in self.contours_slice.iterrows():
            self.rtree.insert("polygon-%d" % cidx, Rect(
                minx=cont['xmin'], miny=cont['ymin'],
                maxx=cont['xmax'], maxy=cont['ymax']))

    # %% =====================================================================

    def set_tree_dict(self):
        """Get tree in convenience dict format (dicts inside dicts)."""
        def _traverse(node):
            """Recursively traverse tree till you get to leafs."""
            if not node.is_leaf():
                node_dict = dict()
                for c in node.children():
                    node_dict[c.index] = _traverse(c)
                return node_dict
            else:
                return node.index

        self.tree_dict = _traverse(self.rtree.cursor)

    # %% =====================================================================

    def set_hierarchy(self):
        """Get hierarchy of node indices."""
        self.hierarchy = dict()

        def _add_hierarchy_level(node_dict, level, parent_idx):
            """Recursively add hierarchy levels."""
            lk = "level-%d" % (level)

            child_nodes = [
                {'nidx': k, 'parent_idx': parent_idx,
                 'is_leaf': type(v) is not dict} for k, v in node_dict.items()]

            if len(child_nodes) < 1:
                return

            # add to current level
            if lk in self.hierarchy.keys():
                self.hierarchy[lk].extend(child_nodes)
            else:
                self.hierarchy[lk] = child_nodes

            # add next level
            for nidx, ndict in node_dict.items():
                if type(ndict) is dict:
                    _add_hierarchy_level(ndict, level=level+1, parent_idx=nidx)

        _add_hierarchy_level(self.tree_dict, level=0, parent_idx=0)

    # %% =====================================================================

    def _merge_polygons(self, poly_list):
        if self.buffer_size > 0:
            poly_list = [j.buffer(self.buffer_size) for j in poly_list]
            merged_polys = cascaded_union(poly_list).buffer(-self.buffer_size)
        else:
            merged_polys = cascaded_union(poly_list)
        return merged_polys

    # %% =====================================================================

    def _merge_leafs(self, leafs):
        nest_polygons = []
        for leaf in leafs:
            leafidx = int(leaf.split('polygon-')[1])
            nest = dict(self.contours_slice.loc[leafidx, :])
            coords = _parse_annot_coords(nest)
            nest_polygons.append(Polygon(coords))
        return self._merge_polygons(nest_polygons)

    # %% =====================================================================

    def _get_merged_polygon(self, nidx):
        self.rtree.cursor._become(nidx)
        leafs = [c.leaf_obj() for c in self.rtree.cursor.children()]
        merged_polygon = self._merge_leafs(leafs)
        return merged_polygon

    # %% =====================================================================

    def get_merged_multipolygon(self):
        """Get final merged shapely multipolygon by hierarchical merger."""
        merged_polygons_all = dict()

        for level in range(len(self.hierarchy) - 1, -1, -1):

            merged_polygons = dict()

            # merge polygons from previous level
            to_merge = dict()
            for node in self.hierarchy["level-%d" % level]:
                if not node['is_leaf']:
                    if node['parent_idx'] not in to_merge.keys():
                        to_merge[node['parent_idx']] = []
                    to_merge[node['parent_idx']].append(merged_polygons_all[
                        "level-%d" % (level + 1)][node['nidx']])
                    del merged_polygons_all[
                        "level-%d" % (level + 1)][node['nidx']]

            for parent_idx, polygon_list in to_merge.items():
                merged_polygons[parent_idx] = self._merge_polygons(
                    polygon_list)

            # merge polygons from this level
            to_merge = dict()
            for node in self.hierarchy["level-%d" % level]:
                if node['is_leaf']:
                    if node['parent_idx'] not in to_merge.keys():
                        to_merge[node['parent_idx']] = []
                    self.rtree.cursor._become(node['nidx'])
                    to_merge[node['parent_idx']].append(
                        self.rtree.cursor.leaf_obj())

            for parent_idx, leafs in to_merge.items():
                merged_polygons[parent_idx] = self._merge_leafs(leafs)

            # assign to persistent dict
            merged_polygons_all['level-%d' % level] = merged_polygons

        return merged_polygons_all['level-0'][0]

    # %% =====================================================================

    def _get_coord_str_from_polygon(self, polygon):
        """Parse shapely polygon coordinates into string form (Internal)."""
        coords = np.int32(polygon.exterior.coords.xy)
        coords_x = ",".join([str(j) for j in coords[0, :]])
        coords_y = ",".join([str(j) for j in coords[1, :]])
        return coords_x, coords_y, coords.T

    # %% =====================================================================

    def _add_single_merged_edge_contour(self, polygon, group):
        """Add single contour to self.new_contours (Internal)."""
        idx = self.new_contours.shape[0]
        self.new_contours.loc[idx, 'type'] = 'polyline'
        self.new_contours.loc[idx, 'group'] = group
        self.new_contours.loc[idx, 'has_holes'] = int(
            polygon.boundary.geom_type == 'MultiLineString')
        coords_x, coords_y, coords = self._get_coord_str_from_polygon(
            polygon)
        self.new_contours.loc[idx, 'coords_x'] = coords_x
        self.new_contours.loc[idx, 'coords_y'] = coords_y
        xmin, ymin = np.min(coords, axis=0)
        xmax, ymax = np.max(coords, axis=0)
        self.new_contours.loc[idx, 'xmin'] = xmin
        self.new_contours.loc[idx, 'ymin'] = ymin
        self.new_contours.loc[idx, 'xmax'] = xmax
        self.new_contours.loc[idx, 'ymax'] = ymax
        self.new_contours.loc[idx, 'bbox_area'] = int(
            (ymax - ymin) * (xmax - xmin))

    # %% =====================================================================

    def _add_merged_multipolygon_contours(
            self, merged_multipolygon, group, monitorPrefix=""):
        """Add merged polygons to self.new_contours df (Internal)."""
        if merged_multipolygon.geom_type == "Polygon":
            merged_multipolygon = [merged_multipolygon, ]
        for pno, polygon in enumerate(merged_multipolygon):
            self._print2("%s: contour %d of %d" % (
                monitorPrefix, pno+1, len(merged_multipolygon)))
            self._add_single_merged_edge_contour(polygon, group=group)

    # %% =====================================================================

    def run_for_single_group(self, group, monitorPrefix=""):
        """Run sequence for merging polygons & adding contours (one group)."""
        # Prep to get polygons
        self._print1("%s: set_contours_slice" % monitorPrefix)
        self.set_contours_slice(group)
        self._print1("%s: create_rtree" % monitorPrefix)
        self.create_rtree()
        self._print1("%s: set_tree_dict" % monitorPrefix)
        self.set_tree_dict()
        self._print1("%s: set_hierarchy" % monitorPrefix)
        self.set_hierarchy()

        # get shapely multipolygon object with merged adjacent contours
        self._print1("%s: get_merged_multipolygon" % monitorPrefix)
        merged_multipolygon = self.get_merged_multipolygon()

        # add contours to new dataframe
        self._print1("%s: _add_merged_multipolygon_contours" % monitorPrefix)
        self._add_merged_multipolygon_contours(
            merged_multipolygon, group=group)

    # %% =====================================================================

    def run(self):
        """Run sequence for merging polygons & adding contours."""
        for group in self.unique_groups:
            monitorPrefix = "%s: %s" % (self.monitorPrefix, group)
            self.run_for_single_group(group, monitorPrefix=monitorPrefix)

# %% ==========================================================================
