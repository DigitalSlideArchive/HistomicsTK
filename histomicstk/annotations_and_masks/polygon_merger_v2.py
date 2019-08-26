# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 16:01:26 2019.

@author: tageldim
"""

import os
import numpy as np
from pandas import DataFrame, concat
# import cv2
from shapely.geometry.polygon import Polygon
from shapely.ops import cascaded_union
# from histomicstk.annotations_and_masks.masks_to_annotations_handler import (
from masks_to_annotations_handler import (
    Conditional_Print, _parse_annot_coords)
from annotation_and_mask_utils import parse_slide_annotations_into_table
from pyrtree.rtree import RTree, Rect

# %% =====================================================================


class Polygon_merger_v2(object):
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
        more_allowed_attr = ['', ]
        allowed_attr = list(default_attr.keys()) + more_allowed_attr
        default_attr.update(kwargs)
        self.__dict__.update(
            (k, v) for k, v in default_attr.items() if k in allowed_attr)

        self.contours_df = contours_df

        # To NOT silently ignore rejected keys
        rejected_keys = set(kwargs.keys()) - set(allowed_attr)
        if rejected_keys:
            raise ValueError(
                "Invalid arguments in constructor:{}".format(rejected_keys))

        # verbosity control
        self.cpr1 = Conditional_Print(verbose=self.verbose == 1)
        self._print1 = self.cpr1._print
        self.cpr2 = Conditional_Print(verbose=self.verbose == 2)
        self._print2 = self.cpr2._print

    # %% =====================================================================

# %% =====================================================================
# %% =====================================================================

# %%===========================================================================
# Constants & prep work
# =============================================================================

import girder_client

APIURL = 'http://candygram.neurology.emory.edu:8080/api/v1/'
SOURCE_SLIDE_ID = '5d5d6910bd4404c6b1f3d893'

gc = girder_client.GirderClient(apiUrl=APIURL)
# gc.authenticate(interactive=True)
gc.authenticate(apiKey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')

# %%===========================================================================

# get and parse slide annotations into dataframe
slide_annotations = gc.get('/annotation/item/' + SOURCE_SLIDE_ID)
contours_df = parse_slide_annotations_into_table(slide_annotations)

# %%===========================================================================

# init polygon merger
pm = Polygon_merger_v2(contours_df, verbose=1)

# %%===========================================================================

pm.contours_df.reset_index(inplace=True, drop=True)
pm.new_contours = DataFrame(columns=pm.contours_df.columns)

group = "mostly_tumor"
pm.contours_slice = pm.contours_df.loc[pm.contours_df.loc[
        :, "group"] == group, :]

# %%===========================================================================

# Add contour bounding boxes to R-tree
rtree = RTree()
for cidx, cont in pm.contours_slice.iterrows():
    rtree.insert("polygon-%d" % cidx, Rect(
            minx=cont['xmin'], miny=cont['ymin'],
            maxx=cont['xmax'], maxy=cont['ymax']))


# %%===========================================================================


def traverse(node):
    """recursively traverse tree till you get to leafs"""
    if not node.is_leaf():
        node_dict = dict()
        for c in node.children():
            node_dict[c.index] = traverse(c)
        return node_dict
    else:
        return node.index


# Get tree in convenience dict format (dicts inside dicts)
rtc = rtree.cursor  # root
tree_dict = traverse(rtc)


# %%===========================================================================

# Get hierarchy of node indices


def _get_hierarchy():
    """Get hierarchy of node indices"""

    hierarchy = dict()

    def _add_hierarchy_level(node_dict, level, parent_idx):

        lk = "level-%d" % (level)

        child_nodes = [
            {'nidx': k, 'parent_idx': parent_idx,
             'is_leaf': type(v) is not dict} for k, v in node_dict.items()]

        if len(child_nodes) < 1:
            return

        # add to current level
        if lk in hierarchy.keys():
            hierarchy[lk].extend(child_nodes)
        else:
            hierarchy[lk] = child_nodes

        # add next level
        for nidx, ndict in node_dict.items():
            if type(ndict) is dict:
                _add_hierarchy_level(ndict, level=level+1, parent_idx=nidx)

    _add_hierarchy_level(tree_dict, level=0, parent_idx=0)

    return hierarchy


hierarchy = _get_hierarchy()

# %%===========================================================================

buffer_size = pm.merge_thresh + 3


def _merge_polygons(poly_list):
    if buffer_size > 0:
        poly_list = [j.buffer(buffer_size) for j in poly_list]
        merged_polys = cascaded_union(poly_list).buffer(-buffer_size)
    else:
        merged_polys = cascaded_union(poly_list)
    return merged_polys


def _merge_leafs(leafs):
    nest_polygons = []
    for leaf in leafs:
        leafidx = int(leaf.split('polygon-')[1])
        nest = dict(pm.contours_slice.loc[leafidx, :])
        coords = _parse_annot_coords(nest)
        nest_polygons.append(Polygon(coords))
    return _merge_polygons(nest_polygons)


def _get_merged_polygon(nidx):
    rtc._become(nidx)
    leafs = [c.leaf_obj() for c in rtc.children()]
    merged_polygon = _merge_leafs(leafs)
    return merged_polygon


def get_merged_multipolygon():

    merged_polygons_all = dict()

    for level in range(len(hierarchy) - 1, -1, -1):

        merged_polygons = dict()

        # merge polygons from previous level
        to_merge = dict()
        for node in hierarchy["level-%d" % level]:
            if not node['is_leaf']:
                if node['parent_idx'] not in to_merge.keys():
                    to_merge[node['parent_idx']] = []
                to_merge[node['parent_idx']].append(merged_polygons_all[
                    "level-%d" % (level + 1)][node['nidx']])
                del merged_polygons_all["level-%d" % (level + 1)][node['nidx']]

        for parent_idx, polygon_list in to_merge.items():
            merged_polygons[parent_idx] = _merge_polygons(polygon_list)

        # merge polygons from this level
        to_merge = dict()
        for node in hierarchy["level-%d" % level]:
            if node['is_leaf']:
                if node['parent_idx'] not in to_merge.keys():
                    to_merge[node['parent_idx']] = []
                rtc._become(node['nidx'])
                to_merge[node['parent_idx']].append(rtc.leaf_obj())

        for parent_idx, leafs in to_merge.items():
            merged_polygons[parent_idx] = _merge_leafs(leafs)

        # assign to persistent dict
        merged_polygons_all['level-%d' % level] = merged_polygons

    return merged_polygons_all['level-0'][0]


# get shapely multipolygon object with merged adjacent contours
merged_multipolygon = get_merged_multipolygon()

# %%===========================================================================

# get new contours database


def _get_coord_str_from_polygon(polygon):
    """Parse shapely polygon coordinates into string form (Internal)."""
    coords = np.int32(polygon.exterior.coords.xy)
    coords_x = ",".join([str(j) for j in coords[0, :]])
    coords_y = ",".join([str(j) for j in coords[1, :]])
    return coords_x, coords_y, coords.T


def _add_single_merged_edge_contour(polygon, group):
    """Add single contour to self.new_contours (Internal)."""
    idx = pm.new_contours.shape[0]
    pm.new_contours.loc[idx, 'type'] = 'polyline'
    pm.new_contours.loc[idx, 'group'] = group
    pm.new_contours.loc[idx, 'has_holes'] = int(
        polygon.boundary.geom_type == 'MultiLineString')
    coords_x, coords_y, coords = _get_coord_str_from_polygon(polygon)
    pm.new_contours.loc[idx, 'coords_x'] = coords_x
    pm.new_contours.loc[idx, 'coords_y'] = coords_y
    xmin, ymin = np.min(coords, axis=0)
    xmax, ymax = np.max(coords, axis=0)
    pm.new_contours.loc[idx, 'xmin'] = xmin
    pm.new_contours.loc[idx, 'ymin'] = ymin
    pm.new_contours.loc[idx, 'xmax'] = xmax
    pm.new_contours.loc[idx, 'ymax'] = ymax
    pm.new_contours.loc[idx, 'bbox_area'] = int(
                (ymax - ymin) * (xmax - xmin))


def _add_merged_multipolygon_contours(
        merged_multipolygon, group, monitorPrefix=""):
    """Add merged polygons to self.new_contours df (Internal)."""
    for pno, polygon in enumerate(merged_multipolygon):
        pm._print2("%s: contour %d of %d" % (
            monitorPrefix, pno+1, len(merged_multipolygon)))
        _add_single_merged_edge_contour(polygon, group=group)

# get new contours dataframe
_add_merged_multipolygon_contours(merged_multipolygon, group=group)


# %%===========================================================================




