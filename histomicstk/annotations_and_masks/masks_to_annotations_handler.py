# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 18:33:48 2019.

@author: tageldim

"""

import numpy as np
from pandas import DataFrame, concat
import cv2
from shapely.geometry.polygon import Polygon
from histomicstk.utils.general_utils import Print_and_log

# %% =====================================================================


def get_contours_from_bin_mask(bin_mask):
    """Given a binary mask, get opencv contours.

    Parameters
    -----------
    bin_mask : nd array
        ground truth mask (m,n) - int32 with [0, 1] values.

    Returns
    --------
    dict
        a dictionary with the following keys:
        - contour group: the actual contour x,y coordinates.
        - heirarchy: contour hierarchy. This contains information about
        how contours relate to each other, in the form:
        [Next, Previous, First_Child, Parent,
        index_relative_to_contour_group]
        The last column is added for convenience and is not part of the
        original opencv output.
        - outer_contours: index of contours that do not have a parent, and
        are therefore the outermost most contours. These may have
        children (holes), however.
        See docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
        for more information.

    """
    # Get contours using openCV. See this for modes:
    # https://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    # we use the flag RETR_CCOMP so that we get boundary as well as holes
    # hierearchy output is: [Next, Previous, First_Child, Parent]
    ROI_cvuint8 = cv2.convertScaleAbs(bin_mask)
    cvout = cv2.findContours(
        ROI_cvuint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if len(cvout) < 3:
        contour_group, hierarchy = cvout[0], cvout[1]
    else:
        contour_group, hierarchy = cvout[1], cvout[2]
    hierarchy = hierarchy[0, ...]
    # outermost contours are those that don't have a parent
    # We'll append an index column to the rightmost end
    # to keep track of things better relative to contour_group, now it is:
    # [Next, Previous, First_Child, Parent, index_relative_to_contour_group]
    hierarchy = np.concatenate((hierarchy, np.arange(
        hierarchy.shape[0])[..., None]), axis=1)
    outer_contours = hierarchy[hierarchy[:, 3] == -1, :]
    conts = {
        'contour_group': contour_group,
        'hierarchy': hierarchy,
        'outer_contours': outer_contours,
    }
    return conts

# %% =====================================================================


def _add_contour_to_df(
        contours_df, mask_shape, conts, cidx, nest_info,
        pad_margin=0, MIN_SIZE=30, MAX_SIZE=None, monitorPrefix=""):
    """Add single contour to dataframe (Internal)."""
    # get coordinates for this contour. These are in x,y format.
    outer_cidx = conts['outer_contours'][cidx, 4]
    cont_outer = conts['contour_group'][outer_cidx][:, 0, :]
    if cont_outer.shape[0] <= 3:
        raise Exception("%s: TOO SIMPLE (%d coordinates) -- IGNORED" % (
            monitorPrefix, cont_outer.shape[0]))

    # Get index of first child (hole)
    inner_cidx = conts['outer_contours'][cidx, 2]
    has_holes = 0 + (inner_cidx > -1)

    # get nest location and size
    xmin, ymin = np.min(cont_outer, axis=0)
    nest_width, nest_height = np.max(
        cont_outer, 0) - np.min(cont_outer, 0)
    ymax = ymin + nest_height
    xmax = xmin + nest_width

    # ignore nests that are too small
    if (nest_height < MIN_SIZE) or (nest_width < MIN_SIZE):
        raise Exception("%s: TOO SMALL (%d x %d pixels) -- IGNORED" % (
            monitorPrefix, nest_height, nest_width))

    # ignore extremely large nests -- THESE MAY CAUSE SEGMENTATION FAULTS
    if MAX_SIZE is not None:
        if (nest_height > MAX_SIZE) or (nest_width > MAX_SIZE):
            raise Exception(
                "%s: EXTREMELY LARGE NEST (%d x %d pixels) -- IGNORED"
                % (monitorPrefix, nest_height, nest_width))

    # assign bounding box location
    ridx = contours_df.shape[0]
    contours_df.loc[ridx, "group"] = nest_info['group']
    contours_df.loc[ridx, "color"] = nest_info['color']
    contours_df.loc[ridx, "ymin"] = ymin - pad_margin
    contours_df.loc[ridx, "ymax"] = ymax - pad_margin
    contours_df.loc[ridx, "xmin"] = xmin - pad_margin
    contours_df.loc[ridx, "xmax"] = xmax - pad_margin

    # add other properties -- maybe useful later
    contours_df.loc[ridx, "has_holes"] = has_holes
    contours_df.loc[ridx, "touches_edge-top"] = 0 + (
        ymin - pad_margin - 2 < 0)
    contours_df.loc[ridx, "touches_edge-left"] = 0 + (
        xmin - pad_margin - 2 < 0)
    contours_df.loc[ridx, "touches_edge-bottom"] = 0 + (
        ymax + pad_margin + 2 > mask_shape[0])
    contours_df.loc[ridx, "touches_edge-right"] = 0 + (
        xmax + pad_margin + 2 > mask_shape[1])

    # get x and y coordinates in HTK friendly format (string)
    cont_outer = conts['contour_group'][outer_cidx][:, 0, :].copy()
    contours_df.loc[ridx, "coords_x"] = \
        ",".join([str(j - pad_margin) for j in list(cont_outer[:, 0])])
    contours_df.loc[ridx, "coords_y"] = \
        ",".join([str(j - pad_margin) for j in list(cont_outer[:, 1])])

    return contours_df

# %% =====================================================================


def _get_contours_df(
        MASK, GTCodes_df, groups_to_get=None, MIN_SIZE=30, MAX_SIZE=None,
        verbose=False, monitorPrefix=""):
    """Parse ground truth mask and gets countours (Internal)."""
    cpr = Print_and_log(verbose=verbose)
    _print = cpr._print

    # pad with zeros to be able to detect edge contours later
    pad_margin = 50
    pad_value = 0
    while (GTCodes_df.GT_code == pad_value).any():
        pad_value += 1
    MASK = np.pad(MASK, pad_margin, 'constant', constant_values=pad_value)

    # Go through unique groups one by one -- each group (i.e. GTCode)
    # is extracted separately by binarizing the multi-class mask
    if groups_to_get is None:
        groups_to_get = list(GTCodes_df.index)
    else:
        groups_to_get = [
            GTCodes_df[GTCodes_df.group == group].head(1).index[0]
            if (GTCodes_df.group == group).any() else group
            for group in groups_to_get]
    contours_df = DataFrame()

    for nestgroup in groups_to_get:

        bin_mask = 0 + (MASK == GTCodes_df.loc[nestgroup, 'GT_code'])

        if bin_mask.sum() < MIN_SIZE * MIN_SIZE:
            _print("%s: %s: NO OBJECTS!!" % (monitorPrefix, nestgroup))
            continue

        _print("%s: %s: getting contours" % (monitorPrefix, nestgroup))
        conts = get_contours_from_bin_mask(bin_mask=bin_mask)
        n_tumor_nests = conts['outer_contours'].shape[0]

        # add nest contours
        _print("%s: %s: adding contours" % (monitorPrefix, nestgroup))
        for cidx in range(n_tumor_nests):
            try:
                nestcountStr = "%s: nest %s of %s" % (
                    monitorPrefix, cidx, n_tumor_nests)
                if cidx % 25 == 100:
                    _print(nestcountStr)

                contours_df = _add_contour_to_df(
                    contours_df, mask_shape=bin_mask.shape,
                    conts=conts, cidx=cidx,
                    nest_info=dict(GTCodes_df.loc[nestgroup, :]),
                    pad_margin=pad_margin, MIN_SIZE=MIN_SIZE,
                    MAX_SIZE=MAX_SIZE, monitorPrefix=nestcountStr)

            except Exception as e:
                _print(e)
                continue

    return contours_df

# %% =====================================================================


def _parse_annot_coords(annot, x_offset=0, y_offset=0):
    """Get x-, y- coordinates in a list format (Internal)."""
    coords_x = [int(j) + x_offset for j in annot['coords_x'].split(',')]
    coords_y = [int(j) + y_offset for j in annot['coords_y'].split(',')]
    coords = [(coords_x[i], coords_y[i]) for i in range(len(coords_x))]
    return coords

# %% =====================================================================


def _discard_nonenclosed_background_group(
        contours_df, background_group='mostly_stroma',
        verbose=False, monitorPrefix=""):
    """If a background group contour is NOT fully enclosed, discard it.

    This is a purely aesthetic method, makes sure that the background group
    contours (eg stroma) are discarded by default to avoid cluttering the
    field when posted to DSA for viewing online. The only exception is
    if they are enclosed within something else (eg tumor), in which case they
    are kept since they represent holes. This is related to
    https://github.com/DigitalSlideArchive/HistomicsTK/issues/675
    (Internal).

    """
    cpr = Print_and_log(verbose=verbose)
    _print = cpr._print

    # isolate background contours and non-background contours with holes
    background = contours_df.loc[
        contours_df.loc[:, "group"] == background_group, :]
    contours_with_holes = contours_df.loc[
        contours_df.loc[:, "group"] != background_group, :]
    contours_with_holes = contours_with_holes.loc[
        contours_with_holes.loc[:, "has_holes"] == 1, :]

    def _append_polygon_if_valid(contDict, cid, polygon_list):
        try:
            polygon = Polygon(_parse_annot_coords(contDict))
            if polygon.is_valid:
                polygon_list.append(polygon)
        except Exception as e:
            _print("%s: contour %d: Shapely Error (below) -- IGNORED!" % (
                monitorPrefix, cid))
            _print(e)
        return polygon_list

    # to avoid redoing things, keep all non-background with holes in a list
    contour_polygons = []
    for cid, cont in contours_with_holes.iterrows():
        contour_polygons = _append_polygon_if_valid(
            dict(cont), cid=cid, polygon_list=contour_polygons)

    # iterate through stromal polygons and find if enclosed within something
    discard_cids = []
    for cid, cont in background.iterrows():
        bck_list = _append_polygon_if_valid(
            dict(cont), cid=cid, polygon_list=[])
        # only keep if enclosed with another contour
        discard = True
        if len(bck_list) > 0:
            for contour_polygon in contour_polygons:
                if contour_polygon.contains(bck_list[0]):
                    discard = False
        if discard:
            discard_cids.append(cid)

    # now drop unnecessary contours
    _print("%s: discarded %d contours" % (monitorPrefix, len(discard_cids)))
    contours_df.drop(discard_cids, axis=0, inplace=True)

    return contours_df


# %% =====================================================================


def get_contours_from_mask(
        MASK, GTCodes_df, groups_to_get=None, MIN_SIZE=30, MAX_SIZE=None,
        get_roi_contour=True, roi_group='roi',
        discard_nonenclosed_background=False, background_group='mostly_stroma',
        verbose=False, monitorPrefix=""):
    """Parse ground truth mask and gets countours for annotations.

    Parameters
    -----------
    MASK : nd array
        ground truth mask (m,n) where pixel values encode group membership.
    GTCodes_df : pandas Dataframe
        the ground truth codes and information dataframe.
        This is a dataframe that is indexed by the annotation group name and
        has the following columns.

        group: str
            group name of annotation, eg. mostly_tumor.
        GT_code: int
            desired ground truth code (in the mask). Pixels of this value
            belong to corresponding group (class).
        color: str
            rgb format. eg. rgb(255,0,0).
    groups_to_get : None
        if None (default) then all groups (ground truth labels) will be
        extracted. Otherwise pass a list of strings like ['mostly_tumor',].
    MIN_SIZE : int
        minimum bounding box size of contour
    MAX_SIZE : None
        if not None, int. Maximum bounding box size of contour. Sometimes
        very large contours cause segmentation faults that originate from
        opencv and are not caught by python, causing the python process
        to unexpectedly hault. If you would like to set a maximum size to
        defend against this, a suggested maximum would be 15000.
    get_roi_contour : bool
        whether to get contour for boundary of region of interest (ROI). This
        is most relevant when dealing with multiple ROIs per slide and with
        rotated rectangular or polygonal ROIs.
    roi_group : str
        name of roi group in the GT_Codes dataframe (eg roi)
    discard_nonenclosed_background : bool
        If a background group contour is NOT fully enclosed, discard it.
        This is a purely aesthetic method, makes sure that the background group
        contours (eg stroma) are discarded by default to avoid cluttering the
        field when posted to DSA for viewing online. The only exception is
        if they are enclosed within something else (eg tumor), in which case
        they are kept since they represent holes. This is related to
        https://github.com/DigitalSlideArchive/HistomicsTK/issues/675
        WARNING - This is a bit slower since the contours will have to be
        converted to shapely polygons. It is not noticeable for hundreds of
        contours, but you will notice the speed difference if you are parsing
        thousands of contours. Default, for this reason, is False.
    background_group : str
        name of background group in the GT_codes dataframe (eg mostly_stroma)
    verbose : bool
        Print progress to screen?
    monitorPrefix : str
        text to prepend to printed statements

    Returns
    --------
    pandas DataFrame
        contours extracted from input mask. The following columns are output.

        group : str
            annotation group (ground truth label).
        color : str
            annotation color if it were to be posted to DSA.
        is_roi : bool
            whether this annotation is a region of interest boundary
        ymin : int
            minimun y coordinate
        ymax : int
            maximum y coordinate
        xmin : int
            minimum x coordinate
        xmax : int
            maximum x coordinate
        has_holes : bool
            whether this contour has holes
        touches_edge-top : bool
            whether this contour touches top mask edge
        touches_edge-bottom : bool
            whether this contour touches bottom mask edge
        touches_edge-left : bool
            whether this contour touches left mask edge
        touches_edge-right : bool
            whether this contour touches right mask edge
        coords_x : str
            vertix x coordinates comma-separated values
        coords_y
            vertix y coordinated comma-separated values

    """
    if MASK.sum() < 3:
        raise Exception("Mask is empty!!")
    cpr = Print_and_log(verbose=verbose)
    _print = cpr._print
    if groups_to_get is not None:
        _print("""WARNING!! Only specify groups_to_get is you do NOT mind
               having NO holes in polygons with holes that are occupied
               by a non-specified group. For example, let's say you
               specified that you only want to extract contours for
               tumor and stroma. If there is a large tumor polygon with two
               holes for stroma and blood vessel, the stroma hole will be
               accounted for, but not the blood vessel hole when you
               post these contours to DSA for viewing then pull them
               to be parse back into mask form. It's a subtle issue related
               to
               https://github.com/DigitalSlideArchive/HistomicsTK/issues/675
               and will eventually be accounted for once HistomicsTK
               has an official format to encode polygons with holes.""")

    cont_kwargs = {
        'GTCodes_df': GTCodes_df,
        'MIN_SIZE': MIN_SIZE,
        'MAX_SIZE': MAX_SIZE,
        'verbose': verbose,
    }

    # get contours df for non-roi contours
    contours_df = _get_contours_df(
        MASK=MASK, groups_to_get=groups_to_get,
        monitorPrefix="%s: %s" % (monitorPrefix, "non-roi"),
        **cont_kwargs)

    # discard non-enclosed background (eg stroma) if needed
    if discard_nonenclosed_background:
        contours_df = _discard_nonenclosed_background_group(
            contours_df, background_group=background_group, verbose=verbose,
            monitorPrefix="%s: %s" % (monitorPrefix, "discarding backgrnd"))

    # get contours df for roi boundary and concat
    if get_roi_contour:
        MASK_BIN = np.zeros(MASK.shape, dtype=np.uint8)
        MASK_BIN[MASK > 0] = GTCodes_df.loc[roi_group, 'GT_code']
        contours_df_roi = _get_contours_df(
            MASK=MASK_BIN, groups_to_get=[roi_group, ],
            monitorPrefix="%s: %s" % (monitorPrefix, roi_group),
            **cont_kwargs)
        contours_df = concat(
            (contours_df_roi, contours_df), axis=0, ignore_index=True)

    return contours_df


# %% =====================================================================


def get_single_annotation_document_from_contours(
        contours_df_slice, docname='default',
        F=1.0, X_OFFSET=0, Y_OFFSET=0, opacity=0.3,
        lineWidth=4.0, verbose=True, monitorPrefix=""):
    """Given dataframe of contours, get annotation document.

    This uses the large_image annotation schema to create an annotation
    document that maybe posted to DSA for viewing using something like:
    resp = gc.post("/annotation?itemId=" + slide_id, json=annotation_doc)
    The annotation schema can be found at:
    github.com/girder/large_image/blob/master/docs/annotations.md .

    Parameters
    -----------
    contours_df_slice : pandas DataFrame
        The following columns are of relevance and must be contained.

        group : str
            annotation group (ground truth label).
        color : str
            annotation color if it were to be posted to DSA.
        coords_x : str
            vertix x coordinates comma-separated values
        coords_y
            vertix y coordinated comma-separated values
    docname : str
        annotation document name
    F : float
        how much smaller is the mask where the contours come from is relative
        to the slide scan magnification. For example, if the mask is at 10x
        whereas the slide scan magnification is 20x, then F would be 2.0.
    X_OFFSET : int
        x offset to add to contours at BASE (SCAN) magnification
    Y_OFFSET : int
        y offset to add to contours at BASE (SCAN) magnification
    opacity : float
        opacity of annotation elements (in the range [0, 1])
    lineWidth : float
        width of boarders of annotation elements
    verbose : bool
        Print progress to screen?
    monitorPrefix : str
        text to prepend to printed statements

    Returns
    --------
    dict
        DSA-style annotation document ready to be post for viewing.

    """
    cpr = Print_and_log(verbose=verbose)
    _print = cpr._print

    def _get_fillColor(lineColor):
        fillColor = lineColor.replace("rgb", "rgba")
        return fillColor[:fillColor.rfind(")")] + ",%.1f)" % opacity

    # Init annotation document in DSA style
    annotation_doc = {'name': docname, 'description': '', 'elements': []}

    # go through nests
    nno = 0
    nnests = contours_df_slice.shape[0]
    for _, nest in contours_df_slice.iterrows():

        nno += 1
        nestStr = "%s: contour %d of %s" % (monitorPrefix, nno, nnests)
        _print(nestStr)

        # Parse coordinates
        try:
            x_coords = F * np.int32(
                [int(j) for j in nest['coords_x'].split(',')]) + X_OFFSET
            y_coords = F * np.int32(
                [int(j) for j in nest['coords_y'].split(',')]) + Y_OFFSET
            zeros = np.zeros(x_coords.shape, dtype=np.int32)
            coords = np.concatenate(
                (x_coords[:, None], y_coords[:, None], zeros[:, None]),
                axis=1)
            coords = coords.tolist()
            coords.append(coords[0])
        except Exception as e:
            _print("%s: ERROR (below) - moving on!!!" % nestStr)
            _print(e)
            continue

        # assign to annotation style. See:
        # github.com/girder/large_image/blob/master/docs/annotations.md
        annotation_style = {
            "group": nest['group'],
            "type": "polyline",
            "lineColor": nest['color'],
            "lineWidth": lineWidth,
            "closed": True,
            "points": coords,
            "label": {'value': nest['label']},
        }
        if opacity > 0:
            annotation_style["fillColor"] = _get_fillColor(nest['color'])

        # append to document
        annotation_doc['elements'].append(annotation_style)

    return annotation_doc

# %% =====================================================================


def get_annotation_documents_from_contours(
        contours_df, separate_docs_by_group=True, annots_per_doc=200,
        annprops=None, docnamePrefix="", verbose=True, monitorPrefix=""):
    """Given dataframe of contours, get list of annotation documents.

    This method parses a dataframe of contours to a list of dictionaries, each
    of which represents and large_image style annotation. This is a wrapper
    that extends the functionality of the method
    get_single_annotation_document_from_contours(), whose docstring should
    be referenced for implementation details and further explanation.

    Parameters
    -----------
    contours_df : pandas DataFrame
        WARNING - This is modified inside the function, so pass a copy.
        This dataframe includes data on contours extracted from input mask
        using get_contours_from_mask(). If you have contours using some other
        method, just make sure the dataframe follows the same schema as the
        output from get_contours_from_mask(). You may find a sample dataframe
        in the repo at
        ./tests/test_files/annotations_and_masks/sample_contours_df.tsv.
        The following columns are relevant for this method.

        group : str
            annotation group (ground truth label).
        color : str
            annotation color if it were to be posted to DSA.
        coords_x : str
            vertix x coordinates comma-separated values
        coords_y
            vertix y coordinated comma-separated values
    separate_docs_by_group : bool
        if set to True, you get one or more annotation documents (dicts)
        for each group (eg tumor) independently.
    annots_per_doc : int
        maximum number of annotation elements (polygons) per dict. The smaller
        this number, the more numerous the annotation documents, but the more
        seamless it is to post this data to the DSA server or to view using the
        HistomicsTK interface since you will be loading smaller chunks of data
        at a time.
    annprops : dict
        properties of annotation elements. Contains the following keys
        F, X_OFFSET, Y_OFFSET, opacity, lineWidth. Refer to
        get_single_annotation_document_from_contours() for details.
    docnamePrefix : str
        test to prepend to annotation document name
    verbose : bool
        Print progress to screen?
    monitorPrefix : str
        text to prepend to printed statements

    Returns
    --------
    list of dicts
        DSA-style annotation document.

    """
    if annprops is None:
        annprops = {
            'F': 1.0,
            'X_OFFSET': 0,
            'Y_OFFSET': 0,
            'opacity': 0,
            'lineWidth': 4.0,
        }
    if separate_docs_by_group:
        contours_df.loc[:, 'doc_group'] = contours_df.loc[:, 'group']
    else:
        contours_df.loc[:, 'doc_group'] = 'default'

    if 'label' not in contours_df.columns:
        contours_df.loc[:, 'label'] = contours_df.loc[:, 'group']

    # Each style goes to separate document(s) if sepate_docs_by_group
    annotation_docs = []
    for doc_group in set(contours_df.loc[:, 'doc_group']):

        # separate annotations with this group
        contours_df_slice = contours_df.loc[
            contours_df.loc[:, 'doc_group'] == doc_group, :]

        # Add every N annotations to a separate document
        if contours_df_slice.shape[0] > annots_per_doc:
            docbounds = list(range(
                0, contours_df_slice.shape[0], annots_per_doc))
            docbounds[-1] = contours_df_slice.shape[0]
        else:
            docbounds = [0, contours_df_slice.shape[0]]

        for docidx in range(len(docbounds)-1):
            docStr = "%s: %s: doc %d of %d" % (
                monitorPrefix, doc_group, docidx+1, len(docbounds)-1)
            start = docbounds[docidx]
            end = docbounds[docidx+1]
            annotation_doc = get_single_annotation_document_from_contours(
                contours_df_slice.iloc[start:end, :],
                docname="%s_%s-%d" % (docnamePrefix, doc_group, docidx),
                verbose=verbose, monitorPrefix=docStr, **annprops)
            if len(annotation_doc['elements']) > 0:
                annotation_docs.append(annotation_doc)

    return annotation_docs

# %% =====================================================================
