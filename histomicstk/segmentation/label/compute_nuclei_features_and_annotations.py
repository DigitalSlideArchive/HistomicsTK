from histomicstk.segmentation import label as htk_label

from ...features.compute_fsd_features import compute_fsd_features
from ...features.compute_gradient_features import compute_gradient_features
from ...features.compute_haralick_features import compute_haralick_features
from ...features.compute_intensity_features import compute_intensity_features
from ...features.compute_morphometry_features import compute_morphometry_features
import histomicstk.segmentation as htk_seg
import numpy as np
import pandas as pd
from skimage.measure import regionprops
from ._trace_object_boundaries_cython import _trace_object_boundaries_cython


def compute_nuclei_features_and_annotations(im_label,tile_info, im_nuclei=None, im_cytoplasm=None,
                            fsd_bnd_pts=128, fsd_freq_bins=6, cyto_width=8,
                            num_glcm_levels=32,
                            morphometry_features_flag=True,
                            fsd_features_flag=True,
                            intensity_features_flag=True,
                            gradient_features_flag=True,
                            haralick_features_flag=True,
                            ):
    

    # sanity checks
    if any([
        intensity_features_flag,
        gradient_features_flag,
        haralick_features_flag,
    ]):
        assert im_nuclei is not None, 'You must provide nuclei intensity!'

    # TODO: this pipeline uses loops a lot. For each set of features it
    #  iterates over all nuclei, which may become an issue when one needs to
    #  do this for lots and lots of slides and 10^6+ nuclei. Consider
    #  improving efficiency in the future somehow (cython? reuse? etc)

    feature_list = []

    # get the objects in im_label
    nuclei_props = regionprops(im_label, intensity_image=im_nuclei)

    # extract object locations and identifiers
    idata = pd.DataFrame()

    #start the data counter
    i = 1
    X = []
    Y = []
    x_start = -1
    y_start = -1
    max_length = float('inf')
    conn = 4
    nuclei_annot_list = []
    gx = tile_info['gx']
    gy = tile_info['gy']
    wfrac = tile_info['gwidth'] / np.double(tile_info['width'])
    hfrac = tile_info['gheight'] / np.double(tile_info['height'])


    for i,rprops in enumerate(nuclei_props):
        # get bounds of label mask
        min_row, min_col, max_row, max_col = rprops.bbox

        # grab label mask
        lmask = (
                im_label[
                    min_row:max_row, min_col:max_col
                ] == rprops.label
            ).astype(bool)
        
        mrows = max_row - min_row + 2
        mcols = max_col - min_col + 2

        mask = np.zeros((mrows, mcols))
        mask[1:mrows - 1, 1:mcols - 1] = lmask

        by, bx = _trace_object_boundaries_cython(
                np.ascontiguousarray(
                    mask, dtype=int), conn, x_start, y_start, max_length
            )

        bx = bx + min_row - 1
        by = by + min_col - 1

        
        bx, by = _remove_thin_colinear_spurs(bx, by,0.01)

        if len(bx) > 0:
                X.append(bx)
                Y.append(by)

                #extract features
                idata.at[i, 'Label'] = rprops.label
                idata.at[i, 'Identifier.Xmin'] = rprops.bbox[1]
                idata.at[i, 'Identifier.Ymin'] = rprops.bbox[0]
                idata.at[i, 'Identifier.Xmax'] = rprops.bbox[3]
                idata.at[i, 'Identifier.Ymax'] = rprops.bbox[2]
                idata.at[i, 'Identifier.CentroidX'] = rprops.centroid[1]
                idata.at[i, 'Identifier.CentroidY'] = rprops.centroid[0]
                if im_nuclei is not None:
                    # intensity-weighted centroid
                    wcy, wcx = rprops.weighted_centroid
                    idata.at[i, 'Identifier.WeightedCentroidX'] = wcx
                    idata.at[i, 'Identifier.WeightedCentroidY'] = wcy
                feature_list.append(idata)

                # compute cytoplasm mask
                if im_cytoplasm is not None:

                    cyto_mask = htk_label.dilate_xor(im_label, neigh_width=cyto_width)

                    cyto_props = regionprops(cyto_mask, intensity_image=im_cytoplasm)

                    # ensure that cytoplasm props order corresponds to the nuclei
                    lablocs = {v['label']: i for i, v in enumerate(cyto_props)}
                    cyto_props = [cyto_props[lablocs[v['label']]] if v['label'] in lablocs else None
                                for v in nuclei_props]

                # compute morphometry features
                if morphometry_features_flag:

                    fmorph = compute_morphometry_features(im_label, rprops=nuclei_props)

                    feature_list.append(fmorph)

                # compute FSD features
                if fsd_features_flag:

                    ffsd = compute_fsd_features(im_label, fsd_bnd_pts, fsd_freq_bins,
                                                cyto_width, rprops=nuclei_props)

                    feature_list.append(ffsd)

                # compute nuclei intensity features
                if intensity_features_flag:

                    fint_nuclei = compute_intensity_features(im_label, im_nuclei,
                                                            rprops=nuclei_props)
                    fint_nuclei.columns = ['Nucleus.' + col
                                        for col in fint_nuclei.columns]

                    feature_list.append(fint_nuclei)

                # compute cytoplasm intensity features
                if intensity_features_flag and im_cytoplasm is not None:

                    fint_cytoplasm = compute_intensity_features(cyto_mask, im_cytoplasm,
                                                                rprops=cyto_props)
                    fint_cytoplasm.columns = ['Cytoplasm.' + col
                                            for col in fint_cytoplasm.columns]

                    feature_list.append(fint_cytoplasm)

                # compute nuclei gradient features
                if gradient_features_flag:

                    fgrad_nuclei = compute_gradient_features(im_label, im_nuclei,
                                                            rprops=nuclei_props)
                    fgrad_nuclei.columns = ['Nucleus.' + col
                                            for col in fgrad_nuclei.columns]

                    feature_list.append(fgrad_nuclei)

                # compute cytoplasm gradient features
                if gradient_features_flag and im_cytoplasm is not None:

                    fgrad_cytoplasm = compute_gradient_features(cyto_mask, im_cytoplasm,
                                                                rprops=cyto_props)
                    fgrad_cytoplasm.columns = ['Cytoplasm.' + col
                                            for col in fgrad_cytoplasm.columns]

                    feature_list.append(fgrad_cytoplasm)

                # compute nuclei haralick features
                if haralick_features_flag:

                    fharalick_nuclei = compute_haralick_features(
                        im_label, im_nuclei,
                        num_levels=num_glcm_levels,
                        rprops=nuclei_props
                    )

                    fharalick_nuclei.columns = ['Nucleus.' + col
                                                for col in fharalick_nuclei.columns]

                    feature_list.append(fharalick_nuclei)

                # compute cytoplasm haralick features
                if haralick_features_flag and im_cytoplasm is not None:

                    fharalick_cytoplasm = compute_haralick_features(
                        cyto_mask, im_cytoplasm,
                        num_levels=num_glcm_levels,
                        rprops=cyto_props
                    )

                    fharalick_cytoplasm.columns = ['Cytoplasm.' + col
                                                for col in fharalick_cytoplasm.columns]

                    feature_list.append(fharalick_cytoplasm)

    print('>>concat all the feature data')

    # Merge all features
    fdata = pd.concat(feature_list, axis=1)

    print('>>concat all the nuclei annotations')

    #merge all nuclei segs
    for j in range(len(X)):
        bx, by = X[j],Y[j]

        for i in range(len(bx)):
            # get boundary points and convert to base pixel space

            num_points = len(bx)

            if num_points < 3:
                continue

            cur_points = np.zeros((num_points, 3))
            cur_points[:, 0] = np.round(gx + bx[i] * wfrac, 2)
            cur_points[:, 1] = np.round(gy + by[i] * hfrac, 2)
            cur_points = cur_points.tolist()

            # create annotation json
            cur_annot = {
                'type': 'polyline',
                'points': cur_points,
                'closed': True,
                'fillColor': 'rgba(0,0,0,0)',
                'lineColor': 'rgb(0,255,0)'
            }

            nuclei_annot_list.append(cur_annot)

    return fdata, nuclei_annot_list


def _remove_thin_colinear_spurs(px, py, eps_colinear_area=0):
    """Simplifies the given list of points by removing colinear spurs
    """

    keep = []  # indices of points to keep

    anchor = -1
    testpos = 0

    while testpos < len(px):

        # get coords of next triplet of points to test
        if testpos == len(px) - 1:
            if not keep:
                break
            nextpos = keep[0]
        else:
            nextpos = testpos + 1

        ind = [anchor, testpos, nextpos]
        x1, x2, x3 = px[ind]
        y1, y2, y3 = py[ind]

        # compute area of triangle formed by triplet
        area = 0.5 * np.linalg.det(
            np.array([[x1, x2, x3], [y1, y2, y3], [1, 1, 1]])
        )

        # if area > cutoff, add testpos to keep and move anchor to testpos
        if abs(area) > eps_colinear_area:

            keep.append(testpos)  # add testpos to keep list
            anchor = testpos      # make testpos the next anchor point
            testpos += 1

        else:

            testpos += 1

    px = px[keep]
    py = py[keep]

    return px, py
