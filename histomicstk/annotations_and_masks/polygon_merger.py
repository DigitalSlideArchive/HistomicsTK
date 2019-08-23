# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 16:25:06 2019

@author: tageldim
"""

import os
import numpy as np
from pandas import DataFrame, concat
# import cv2
from shapely.geometry.polygon import Polygon
from shapely.ops import cascaded_union
from PIL import Image
from imageio import imread
from masks_to_annotations_handler import (
    Conditional_Print, get_contours_from_mask, _parse_annot_coords,
    _discard_nonenclosed_background_group)


# %% =====================================================================

class Polygon_merger(object):
    """Methods to merge polygons in tiled masks."""

    def __init__(self, maskpaths, GTCodes_df, **kwargs):
        """Init Polygon_merger object.

        Arguments:
        -----------
        maskpaths : list
            list of strings representing pathos to masks
        GTCodes_df : pandas DataFrame
            the ground truth codes and information dataframe.
            This is a dataframe that is indexed by the annotation group name
            and has the following columns.

            group: str
                group name of annotation, eg. mostly_tumor.
            GT_code: int
                desired ground truth code (in the mask). Pixels of this value
                belong to corresponding group (class).
            coords_x : str
                vertix x coordinates comma-separated values
            coords_y
                vertix y coordinated comma-separated values
            color: str
                rgb format. eg. rgb(255,0,0).
        merge_thresh : int
            how close do the polygons need to be (in pixels) to be merged
        contkwargs : dict
            dictionary of kwargs to pass to get_contours_from_mask()
        verbose : bool
            Print progress to screen?
        monitorPrefix : str
            text to prepend to printed statements
        """

        self.maskpaths = maskpaths
        self.GTCodes_df = GTCodes_df

        # see: https://stackoverflow.com/questions/8187082/how-can-you-set-...
        # class-attributes-from-variable-arguments-kwargs-in-python
        default_attr = {
            'verbose': True,
            'monitorPrefix': "",
            'merge_thresh': 3,
            'contkwargs': {
                'GTCodes_df': GTCodes_df,
                'get_roi_contour': False,  # important
                'discard_nonenclosed_background': False,  # important
                'MIN_SIZE': 2,
                'MAX_SIZE': None,
                'verbose': True,
            },
        }
        more_allowed_attr = ['', ]
        allowed_attr = list(default_attr.keys()) + more_allowed_attr
        default_attr.update(kwargs)
        self.__dict__.update(
            (k, v) for k, v in default_attr.items() if k in allowed_attr)

        # To NOT silently ignore rejected keys
        rejected_keys = set(kwargs.keys()) - set(allowed_attr)
        if rejected_keys:
            raise ValueError(
                "Invalid arguments in constructor:{}".format(rejected_keys))

        # some sanity checks
        assert not (
            self.contkwargs['get_roi_contour']
            or self.contkwargs['discard_nonenclosed_background'])

        self.cpr = Conditional_Print(verbose=self.verbose)
        self._print = self.cpr._print

    # %% =====================================================================

    def get_contours_from_all_masks(self, monitorPrefix=""):
        """Get contours_df from all masks.

        This is a wrapper around get_contours_from_mask(), with the added
        functionality of separating out contorus at roi edge from those that
        are not.

        Sets:
        ---------
        self.ordinary_contours : dict
            indexed by maskname, each entry is a contours dataframe
        self.edge_contours : dict
            indexed by maskname, each entry is a contours dataframe
        self.merged_contours : pandas DataFrame
            single dataframe to save all merged contours

        """
        ordinary_contours = dict()
        edge_contours = dict()

        for midx, maskpath in enumerate(self.maskpaths):

            # extract contours
            MASK = imread(maskpath)
            contours_df = get_contours_from_mask(
                MASK=MASK,
                monitorPrefix="%s: mask %d of %d" % (
                    monitorPrefix, midx, len(self.maskpaths)),
                **self.contkwargs)

            # separate edge from non-edge contours
            edgeids = []
            for edge in ['top', 'left', 'bottom', 'right']:
                edgeids.extend(list(contours_df.loc[contours_df.loc[
                    :, 'touches_edge-%s' % edge] == 1, :].index))
            edgeids = list(set(edgeids))
            roiname = os.path.split(maskpath)[1]
            edge_contours[roiname] = contours_df.loc[edgeids, :].copy()
            ordinary_contours[roiname] = contours_df.drop(edgeids, axis=0)

        self.ordinary_contours = ordinary_contours
        self.edge_contours = edge_contours
        # init dataframe to save merged contours
        colnames = edge_contours[list(edge_contours.keys())[0]].columns
        self.merged_contours = DataFrame(columns=colnames)

    # %% =====================================================================

    def _get_mask_offsets_from_masknames(self):
        """Get dictionary of mask offsets (top and left) (Internal).

        The pattern _left-123_ and _top-123_ is assumed to
        encode the x and y offset of the mask at base magnification.

        Returns:
        ----------
        dict
            indexed by maskname, each entry is a dict with keys top and left.

        """
        roi_offsets = dict()
        for maskpath in self.maskpaths:
            maskname = os.path.split(maskpath)[1]
            roi_offsets[maskname] = {
                'left': int(maskname.split('_left-')[1].split('_')[0]),
                'top': int(maskname.split('_top-')[1].split('_')[0]),
            }
        return roi_offsets

    # %% =====================================================================

    def get_roi_bboxes(self, roi_offsets=None):
        """Get dictionary of roi bounding boxes.

        Arguments:
        -----------
        roi_offsets : dict (default, None)
            dict indexed by maskname, each entry is a dict with keys
            top and left each is an integer. If None, then the pattern
            _left-123_ and _top-123_ is assumed to encode the x and y
            offset of the mask (i.e. inferred from mask name)

        Sets:
        ----------
        self.roiinfos : dict
            dict indexed by maskname, each entry is a dict with keys
            top, left, bottom, right, all of which are integers

        """
        if roi_offsets is not None:
            roiinfos = roi_offsets.copy()
        else:
            # get offset for all rois. This result is a dict that is indexed
            # by maskname, each entry is a dict with keys 'top' and 'left'.
            roiinfos = self._get_mask_offsets_from_masknames()

        for maskpath in self.maskpaths:
            # Note: the following method does NOT actually load the mask
            # but just uses pillow to get its metadata. See:
            # https://stackoverflow.com/questions/15800704/ ...
            # ... get-image-size-without-loading-image-into-memory
            mask_obj = Image.open(maskpath, mode='r')
            width, height = mask_obj.size
            maskname = os.path.split(maskpath)[1]
            roiinfos[maskname]['right'] = roiinfos[maskname]['left'] + width
            roiinfos[maskname]['bottom'] = roiinfos[maskname]['top'] + height

        self.roiinfos = roiinfos

    # %% =====================================================================

    def _get_roi_pairs(self):
        """Get unique roi pairs (Internal)."""
        ut = np.triu_indices(len(self.roinames), k=1)
        roi_pairs = []
        for pairidx in range(len(ut[0])):
            roi_pairs.append((ut[0][pairidx], ut[1][pairidx]))
        return roi_pairs

    # %% =====================================================================

    def _get_shared_roi_edges(self):
        """Get shared edges between rois in same slide (Internal)."""
        self.roinames = list(self.roiinfos.keys())
        edgepairs = [
            ('left', 'right'), ('right', 'left'),
            ('top', 'bottom'), ('bottom', 'top'),
        ]
        roi_pairs = self._get_roi_pairs()

        # init shared edges
        shared_edges = DataFrame(columns=[
                'roi1-name', 'roi1-edge', 'roi2-name', 'roi2-edge'])

        for roi_pair in roi_pairs:
            roi1name = self.roinames[roi_pair[0]]
            roi2name = self.roinames[roi_pair[1]]
            idx = shared_edges.shape[0]
            for edgepair in edgepairs:
                # check if they share bounds for one edge
                if np.abs(
                      self.roiinfos[roi1name][edgepair[0]]
                      - self.roiinfos[roi2name][edgepair[1]]) < 2:
                    # ensure they overlap in location along other axis
                    if 'left' in edgepair:
                        start, end = ('top', 'bottom')
                    else:
                        start, end = ('left', 'right')
                    realStart = np.min(
                        (self.roiinfos[roi1name][start],
                         self.roiinfos[roi2name][start]))
                    realEnd = np.max(
                        (self.roiinfos[roi1name][end],
                         self.roiinfos[roi2name][end]))
                    length = realEnd - realStart
                    nonoverlap_length = (
                        self.roiinfos[roi1name][end]
                        - self.roiinfos[roi1name][start]) + (
                        self.roiinfos[roi2name][end]
                        - self.roiinfos[roi2name][start])
                    if length < nonoverlap_length:
                        shared_edges.loc[idx, 'roi1-name'] = roi1name
                        shared_edges.loc[idx, 'roi1-edge'] = edgepair[0]
                        shared_edges.loc[idx, 'roi2-name'] = roi2name
                        shared_edges.loc[idx, 'roi2-edge'] = edgepair[1]

        self.shared_edges = shared_edges

    # %% =====================================================================

    def _get_merge_pairs(self, edgepair, group, monitorPrefix=""):
        """Get nest dataframes and indices of which ones to merge."""

        def _get_nests_slice(ridx=1):
            Nests = self.edge_contours[edgepair['roi%d-name' % ridx]]
            edgevar = "touches_edge-%s" % (edgepair['roi%d-edge' % ridx])
            Nests = Nests.loc[Nests.loc[:, edgevar] == 1, :]
            Nests = Nests.loc[Nests.loc[:, 'group'] == group, :]
            return Nests

        # Nests of the same label. The nest IDs are using the index of the
        # roi dataframe from the edge_nests dictionary
        Nests1 = _get_nests_slice(1)
        Nests2 = _get_nests_slice(2)

        # to avoid redoing things, keep all polygons in a list
        polygons1 = []
        nno1 = 0
        nno1Max = Nests1.shape[0]
        for nid1, nest1 in Nests1.iterrows():
            nno1 += 1
            self._print("%s: edge1-nest %d of %d" % (
                monitorPrefix, nno1, nno1Max))
            try:
                coords = np.array(_parse_annot_coords(nest1))
                coords[:, 0] = coords[:, 0] + self.roiinfos[
                    edgepair['roi1-name']]['left']
                coords[:, 1] = coords[:, 1] + self.roiinfos[
                    edgepair['roi1-name']]['top']
                polygons1.append((nid1, Polygon(coords)))
            except Exception as e:
                self._print(
                    "%s: edge1-nest %d of %d: Shapely Error (below)" % (
                        monitorPrefix, nno1, nno1Max))
                self._print(e)

        # go through the "other" polygons to get merge list
        to_merge = DataFrame(columns=[
            'nest1-roiname', 'nest1-nid', 'nest2-roiname', 'nest2-nid'])
        nno2 = 0
        nno2Max = Nests2.shape[0]
        for nid2, nest2 in Nests2.iterrows():
            nno2 += 1
            try:
                coords = np.array(_parse_annot_coords(nest2))
                coords[:, 0] = coords[:, 0] + self.roiinfos[
                    edgepair['roi2-name']]['left']
                coords[:, 1] = coords[:, 1] + self.roiinfos[
                    edgepair['roi2-name']]['top']
                polygon2 = Polygon(coords)
            except Exception as e:
                self._print(
                    "%s: edge2-nest %d of %d: Shapely Error (below)" % (
                        monitorPrefix, nno2, nno2Max))
                self._print(e)
                continue

            nno1Max = len(polygons1)-1
            for nno1, poly1 in enumerate(polygons1):
                self._print(
                    "%s: edge2-nest %d of %d: vs. edge1-nest %d of %d" % (
                        monitorPrefix, nno2, nno2Max, nno1+1, nno1Max+1))
                nid1, polygon1 = poly1
                if polygon1.distance(polygon2) < self.merge_thresh:
                    idx = to_merge.shape[0]
                    to_merge.loc[idx, 'nest1-roiname'] = edgepair['roi1-name']
                    to_merge.loc[idx, 'nest1-nid'] = nid1
                    to_merge.loc[idx, 'nest2-roiname'] = edgepair['roi2-name']
                    to_merge.loc[idx, 'nest2-nid'] = nid2

        return to_merge

    # %% =====================================================================

    def _get_merge_df(self, group, monitorPrefix=""):
        """Get merge dataframe (pairs along shared edges)."""
        merge_df = DataFrame()
        for rpno, edgepair in self.shared_edges.iterrows():
            edgepair = dict(edgepair)
            to_merge = self._get_merge_pairs(
                edgepair=edgepair, group=group,
                monitorPrefix="%s: pair %d of %d" % (
                    monitorPrefix, rpno + 1, self.shared_edges.shape[0]))
            merge_df = concat((merge_df, to_merge), axis=0, ignore_index=True)
        return merge_df

    # %% =====================================================================

    def _get_merge_clusters_from_df(self, merge_df, monitorPrefix=""):
        """Assigned each nest to one cluster.

        That is, such that all nests that are connected to each other are
        in the same cluster. Uses hierarchical-like clustering to do this.

        """
        checksum_ref = []

        # initi merge groups with clusters of size 2
        merge_groups = {'level-0': []}
        for _, merge in merge_df.iterrows():
            nname1 = "%s_nid-%d" % (merge['nest1-roiname'], merge['nest1-nid'])
            nname2 = "%s_nid-%d" % (merge['nest2-roiname'], merge['nest2-nid'])
            merge_groups['level-0'].append([nname1, nname2])
            checksum_ref.extend([nname1, nname2])
        checksum_ref = len(list(set(checksum_ref)))

        # go through levels
        level = 0
        keep_going = True
        while keep_going:
            self._print("%s: level %d" % (monitorPrefix, level))
            merge_groups['level-%d' % (level+1)] = []
            reference = merge_groups['level-%d' % (level)].copy()
            # compare each cluster to the others in current level
            for gid, mgroup in enumerate(merge_groups['level-%d' % (level)]):
                for gid2, mgroup2 in enumerate(reference):
                    mgroupSet = set(mgroup)
                    mgroup2Set = set(mgroup2)
                    # if there's anything in common between the two clusters
                    # (assuming they're not the same cluster of course)
                    # then merge them and move them to the next level in the
                    # hierarchy and remove them both from this level
                    if (gid != gid2) and (
                            len(mgroupSet.intersection(mgroup2Set)) > 0):
                        merge_groups['level-%d' % (level+1)].append(
                                list(mgroupSet.union(set(mgroup2Set))))
                        reference[gid] = []
                        reference[gid2] = []
            # cleanup
            merge_groups['level-%d' % (level)] = [
                    j for j in reference if len(j) > 0]
            if len(merge_groups['level-%d' % (level+1)]) < 1:
                del merge_groups['level-%d' % (level+1)]
                keep_going = False
            level += 1

        # now just concatenate all hierarchical levels together
        merge_clusters = []
        for _, mg in merge_groups.items():
            if len(mg) > 0:
                merge_clusters.extend(mg)

        # sanity check
        checksum = np.sum([len(j) for j in merge_clusters])
        assert checksum == checksum_ref, \
            "checksum fail! not every value is assigned exactly one cluster."

        # restructure into dicts with roi name, nest id keys for convenience
        def _parse_to_dict(text):
            parts = tuple(text.split('_nid-'))
            return {'roiname': parts[0], 'nid': int(parts[1])}
        merge_clusters = [
            [_parse_to_dict(j) for j in cl] for cl in merge_clusters]

        return merge_clusters

    # %% =====================================================================

    def _get_merged_polygon(self, cluster):
        """Merges polygons using shapely.

        Given a single cluster from _get_merge_clusters_from_df(), This creates
        and merges polygons into a single cascaded union. It first dilates the
        polygons by buffer_size pixels to make them overlap, merges them,
        then erodes back by buffer_size to get the merged polygon.

        """
        buffer_size = self.merge_thresh + 3
        nest_polygons = []
        for nestinfo in cluster:
            nest = dict(self.edge_contours[nestinfo['roiname']].loc[
                    nestinfo['nid'], :])
            roitop = self.roiinfos[nestinfo['roiname']]['top']
            roileft = self.roiinfos[nestinfo['roiname']]['left']
            coords = _parse_annot_coords(
                nest, x_offset=roileft, y_offset=roitop)
            nest_polygons.append(Polygon(coords).buffer(buffer_size))
        merged_polygon = cascaded_union(nest_polygons).buffer(-buffer_size)
        return merged_polygon

    # %% =====================================================================

    def _get_all_merged_polygons(self, merge_clusters, monitorPrefix=""):
        """Merges polygons using shapely.

        Given a a list of clusters from _get_merge_clusters_from_df(). Creates
        and merges polygons into a single cascaded union. It first dilates the
        polygons by buffer_size pixels to make them overlap, merges them,
        then erodes back by buffer_size to get the merged polygon.

        """
        merged_polygons = []
        for cid, cl in enumerate(merge_clusters):
            self._print("%s: cluster %d of %d" % (
                monitorPrefix, cid+1, len(merge_clusters)))
            merged_polygon = self._get_merged_polygon(cluster=cl)
            merged_polygons.append(merged_polygon)
        return merged_polygons

    # %% =====================================================================

    def _get_coord_str_from_polygon(self, polygon):
        """Parse shapely polygon coordinates into string form (Internal)."""
        # outer_contour_idx = np.argmax([j.length for j in polygon.boundary])
        # outer_contour = polygon.boundary[outer_contour_idx]
        # coords = np.int32(outer_contour.coords.xy)
        coords = np.int32(polygon.exterior.coords.xy)
        coords_x = ",".join([str(j) for j in coords[0, :]])
        coords_y = ",".join([str(j) for j in coords[1, :]])
        return coords_x, coords_y

    # %% =====================================================================

    def _add_single_merged_edge_contour(self, polygon):
        idx = self.merged_contours.shape[0]
        self.merged_contours.loc[idx, 'group'] = group
        self.merged_contours.loc[idx, 'color'] = self.GTCodes_df.loc[
            group, 'color']
        coords_x, coords_y = self._get_coord_str_from_polygon(polygon)
        self.merged_contours.loc[idx, 'coords_x'] = coords_x
        self.merged_contours.loc[idx, 'coords_y'] = coords_y

    # %% =====================================================================

    def _add_merged_edge_contours(
            self, merged_polygons, group, monitorPrefix=""):
        """Add merged polygons to self.merged_contours dataframe."""
        for pno, geometry in enumerate(merged_polygons):
            self._print("%s: contour %d of %d" % (
                monitorPrefix, pno+1, len(merged_polygons)))
            if geometry.type == 'MultiPolygon':
                for polygon in geometry:
                    self._add_single_merged_edge_contour(polygon)
            else:
                self._add_single_merged_edge_contour(geometry)

    # %% =====================================================================

    def _drop_merged_edge_contours(self, merge_df):
        """Drop edge contours that have already been merged (Internal)."""
        for roiname, edge_df in self.edge_contours.items():
            nids = []
            for rid in [1, 2]:
                neststr = 'nest%d' % rid
                nids.extend(list(merge_df.loc[merge_df.loc[
                    :, neststr + '-roiname'] == roiname, neststr + '-nid']))
            self.edge_contours[roiname].drop(nids, axis=0, inplace=True)

    # %% =====================================================================

    def _add_roi_offset_to_contours(self, roi_df, roiname):
        """add roi offset to coordinates of polygons."""
        for idx, annot in roi_df.iterrows():
            coords = np.int32(_parse_annot_coords(dict(annot)))
            coords[:, 0] = coords[:, 0] + self.roiinfos[roiname]['left']
            coords[:, 1] = coords[:, 1] + self.roiinfos[roiname]['top']
            roi_df.loc[idx, 'coords_x'] = ",".join(
                [str(j) for j in coords[:, 0]])
            roi_df.loc[idx, 'coords_y'] = ",".join(
                [str(j) for j in coords[:, 1]])
        return roi_df

    # %% =====================================================================

    def get_concatenated_contours(self):
        """Get concatenated contours and overall bounding box."""
        # concatenate all contours
        all_contours = self.merged_contours.copy()
        for contours_dict in [self.edge_contours, self.ordinary_contours]:
            for roiname, roi_df in contours_dict.items():
                roi_df = self._add_roi_offset_to_contours(
                    roi_df=roi_df, roiname=roiname)
                all_contours = concat(
                    (all_contours, roi_df), axis=0, ignore_index=True)

        # add overall bounding box contour
        idx = all_contours.shape[0]
        top = str(int(np.min([j['top'] for _, j in self.roiinfos.items()])))
        bottom = str(int(
            np.max([j['bottom'] for _, j in self.roiinfos.items()])))
        left = str(int(np.min([j['left'] for _, j in self.roiinfos.items()])))
        right = str(int(
            np.max([j['right'] for _, j in self.roiinfos.items()])))
        all_contours.loc[idx, 'group'] = 'roi'
        all_contours.loc[idx, 'color'] = self.GTCodes_df.loc['roi', 'color']
        all_contours.loc[idx, 'coords_x'] = ",".join(
                [left, right, right, left, left])
        all_contours.loc[idx, 'coords_y'] = ",".join(
                [top, top, bottom, bottom, top])

        return all_contours


# %%===========================================================================
# Constants & prep work
# =============================================================================


import girder_client
from pandas import read_csv

APIURL = 'http://candygram.neurology.emory.edu:8080/api/v1/'
SAMPLE_SLIDE_ID = '5d586d76bd4404c6b1f286ae'

gc = girder_client.GirderClient(apiUrl=APIURL)
# gc.authenticate(interactive=True)
gc.authenticate(apiKey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')

# read GTCodes dataframe
PTESTS_PATH = os.path.join(os.getcwd(), '..', '..', 'plugin_tests')
GTCODE_PATH = os.path.join(PTESTS_PATH, 'test_files', 'sample_GTcodes.csv')
GTCodes_df = read_csv(GTCODE_PATH)
GTCodes_df.index = GTCodes_df.loc[:, 'group']

# This is where masks for adjacent rois are saved
MASK_LOADPATH = os.path.join(
    PTESTS_PATH, 'test_files', 'polygon_merger_roi_masks')
maskpaths = [
    os.path.join(MASK_LOADPATH, j) for j in os.listdir(MASK_LOADPATH)
    if j.endswith('.png')]

# %%===========================================================================

pm = Polygon_merger(maskpaths=maskpaths, GTCodes_df=GTCodes_df)
pm.get_contours_from_all_masks()
pm.get_roi_bboxes()
pm._get_shared_roi_edges()

# %%
for group in pm.GTCodes_df.index:

    monitorPrefix = "%s: %s" % (pm.monitorPrefix, group)

    # get pairs of contours to merge
    merge_df = pm._get_merge_df(
        group=group, monitorPrefix="%s: _get_merge_df" % monitorPrefix)

    # get clusters of polygons to merge
    merge_clusters = pm._get_merge_clusters_from_df(
        merge_df=merge_df,
        monitorPrefix="%s: _get_merge_clusters_from_df" % monitorPrefix)

    # fetch merged polygons
    merged_polygons = pm._get_all_merged_polygons(
        merge_clusters=merge_clusters,
        monitorPrefix="%s: _get_all_merged_polygons" % monitorPrefix)

    # add medged contours to dataframe
    pm._add_merged_edge_contours(
        merged_polygons=merged_polygons, group=group,
        monitorPrefix="%s: _add_merged_edge_contours" % monitorPrefix)

    # drop merged edge contours from edge dataframes
    pm._print("%s: _drop_merged_edge_contours" % monitorPrefix)
    pm._drop_merged_edge_contours(merge_df=merge_df)

# %%

all_contours = pm.get_concatenated_contours()

# %%

from masks_to_annotations_handler import get_annotation_documents_from_contours

# deleting existing annotations in target slide (if any)
existing_annotations = gc.get('/annotation/item/' + SAMPLE_SLIDE_ID)
for ann in existing_annotations:
    gc.delete('/annotation/%s' % ann['_id'])

# get list of annotation documents
annotation_docs = get_annotation_documents_from_contours(
    all_contours.copy(), separate_docs_by_group=True,
    docnamePrefix='test',
    verbose=False, monitorPrefix=SAMPLE_SLIDE_ID + ": annotation docs")

# post annotations to slide -- make sure it posts without errors
for annotation_doc in annotation_docs:
    resp = gc.post(
        "/annotation?itemId=" + SAMPLE_SLIDE_ID, json=annotation_doc)



