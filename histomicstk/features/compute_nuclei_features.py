from histomicstk.cli import utils as cli_utils
from histomicstk.segmentation import label as htk_label

from .compute_fsd_features import compute_fsd_features
from .compute_gradient_features import compute_gradient_features
from .compute_haralick_features import compute_haralick_features
from .compute_intensity_features import compute_intensity_features
from .compute_morphometry_features import compute_morphometry_features


def compute_nuclei_features(im_label, im_nuclei=None, im_cytoplasm=None,
                            fsd_bnd_pts=128, fsd_freq_bins=6, cyto_width=8,
                            num_glcm_levels=32,
                            morphometry_features_flag=True,
                            fsd_features_flag=True,
                            intensity_features_flag=True,
                            gradient_features_flag=True,
                            haralick_features_flag=True,
                            tile_info=None,
                            im_nuclei_seg_mask=None,
                            format=None,
                            return_nuclei_annotation=False,
                            ):
    """
    Calculates features for nuclei classification

    Parameters
    ----------
    im_label : array_like
        A labeled mask image wherein intensity of a pixel is the ID of the
        object it belongs to. Non-zero values are considered to be foreground
        objects.

    im_nuclei : array_like
        Nucleus channel intensity image.

    im_cytoplasm : array_like
        Cytoplasm channel intensity image.

    fsd_bnd_pts : int, optional
        Number of points for boundary resampling to calculate fourier
        descriptors. Default value = 128.

    fsd_freq_bins : int, optional
        Number of frequency bins for calculating FSDs. Default value = 6.

    cyto_width : float, optional
        Estimated width of the ring-like neighborhood region around each
        nucleus to be considered as its cytoplasm. Default value = 8.

    num_glcm_levels: int, optional
        An integer specifying the number of gray levels For example, if
        `NumLevels` is 32,  the intensity values of the input image are
        scaled so they are integers between 0 and 31.  The number of gray
        levels determines the size of the gray-level co-occurrence matrix.

        Default: 32

    morphometry_features_flag : bool, optional
        A flag that can be used to specify whether or not to compute
        morphometry (size and shape) features.
        See histomicstk.features.compute_morphometry_features for more details.

    fsd_features_flag : bool, optional
        A flag that can be used to specify whether or not to compute
        Fouried shape descriptor (FSD) features.
        See `histomicstk.features.compute_fsd_features` for more details.

    intensity_features_flag : bool, optional
        A flag that can be used to specify whether or not to compute
        intensity features from the nucleus and cytoplasm channels.
        See `histomicstk.features.compute_fsd_features` for more details.

    gradient_features_flag : bool, optional
        A flag that can be used to specify whether or not to compute
        gradient/edge features from intensity and cytoplasm channels.
        See `histomicstk.features.compute_gradient_features` for more details.

    haralick_features_flag : bool, optional
        A flag that can be used to specify whether or not to compute
        haralick features from intensity and cytoplasm channels.
        See `histomicstk.features.compute_haralick_features` for more details.

    return_nuclei_annotation :  bool, optional
        Returns the nuclei annotation if kept True

    Returns
    -------
    fdata : pandas.DataFrame
        A pandas data frame containing the features listed below for each
        object/label
    nuclei_annot_list : List
        List containing the boundaries of segmented nuclei in the input image.

    Notes
    -----
    List of features computed by this function

    Identifier
        Location of the nucleus and its code in the input labeled mask.
        Columns are prefixed by *Identifier.*. These include ...

        Identifier.Label (int) - nucleus label in the input labeled mask

        Identifier.Xmin (int) - Left bound

        Identifier.Ymin (int) - Upper bound

        Identifier.Xmax (int) - Right bound

        Identifier.Ymax (int) - Lower bound

        Identifier.CentroidX (float) - X centroid (columns)

        Identifier.CentroidY (float) - Y centroid (rows)

        Identifier.WeightedCentroidX (float) - intensity-weighted X centroid

        Identifier.WeightedCentroidY (float) - intensity-weighted Y centroid

    Morphometry (size, shape, and orientation) features of the nuclei
        See histomicstk.features.compute_morphometry_features for more details.
        Feature names prefixed by *Size.*, *Shape.*, or *Orientation.*.

    Fourier shape descriptor features
        See `histomicstk.features.compute_fsd_features` for more details.
        Feature names are prefixed by *FSD*.

    Intensity features for the nucleus and cytoplasm channels
        See `histomicstk.features.compute_fsd_features` for more details.
        Feature names are prefixed by *Nucleus.Intensity.* for nucleus features
        and *Cytoplasm.Intensity.* for cytoplasm features.

    Gradient/edge features for the nucleus and cytoplasm channels
        See `histomicstk.features.compute_gradient_features` for more details.
        Feature names are prefixed by *Nucleus.Gradient.* for nucleus features
        and *Cytoplasm.Gradient.* for cytoplasm features.

    Haralick features for the nucleus and cytoplasm channels
        See `histomicstk.features.compute_haralick_features` for more details.
        Feature names are prefixed by *Nucleus.Haralick.* for nucleus features
        and *Cytoplasm.Haralick.* for cytoplasm features.

    See Also
    --------
    histomicstk.features.compute_morphometry_features,
    histomicstk.features.compute_fsd_features,
    histomicstk.features.compute_intensity_features,
    histomicstk.features.compute_gradient_features,
    histomicstk.features.compute_haralick_features

    """
    import pandas as pd
    from skimage.measure import regionprops

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

    im_nuclei_bool = im_nuclei is not None

    data = []

    def process_nucleus(nprop, im_nuclei_bool):
        for nprop in nuclei_props:
            row = {
                'Label': nprop.label,
                'Identifier.Xmin': nprop.bbox[1],
                'Identifier.Ymin': nprop.bbox[0],
                'Identifier.Xmax': nprop.bbox[3],
                'Identifier.Ymax': nprop.bbox[2],
                'Identifier.CentroidX': nprop.centroid[1],
                'Identifier.CentroidY': nprop.centroid[0],
            }
            if im_nuclei_bool:
                wcy, wcx = nprop.weighted_centroid
                row['Identifier.WeightedCentroidX'] = wcx
                row['Identifier.WeightedCentroidY'] = wcy
            data.append(row)
        return pd.DataFrame(data)

    # create the DataFrame in one step
    idata = process_nucleus(nuclei_props, im_nuclei_bool)
    feature_list.append(idata)

    def conditional(flag, func, args, kwargs, prefix=None):
        if flag:
            output = func(*args, **kwargs)
            if prefix:
                output.columns = [prefix + col for col in output.columns]
            return output
        else:
            return pd.DataFrame()  # return an empty DataFrame if condition is not met

    # compute cytoplasm mask
    if im_cytoplasm is not None:

        cyto_mask = htk_label.dilate_xor(im_label, neigh_width=cyto_width)

        cyto_props = regionprops(cyto_mask, intensity_image=im_cytoplasm)

        # ensure that cytoplasm props order corresponds to the nuclei
        lablocs = {v['label']: i for i, v in enumerate(cyto_props)}
        cyto_props = [cyto_props[lablocs[v['label']]] if v['label'] in lablocs else None
                      for v in nuclei_props]

    # compute morphometry features
    feature_list.append(conditional(
        morphometry_features_flag,
        compute_morphometry_features,
        [im_label], {'rprops': nuclei_props},
    ))

    # compute FSD features
    feature_list.append(conditional(
        fsd_features_flag,
        compute_fsd_features,
        [im_label, fsd_bnd_pts, fsd_freq_bins, cyto_width], {'rprops': nuclei_props},
    ))

    # compute nuclei intensity features
    feature_list.append(conditional(
        intensity_features_flag,
        compute_intensity_features,
        [im_label, im_nuclei], {'rprops': nuclei_props}, prefix='Nucleus.',
    ))

    # compute cytoplasm intensity features
    if im_cytoplasm is not None:
        feature_list.append(conditional(
            intensity_features_flag,
            compute_intensity_features,
            [cyto_mask, im_cytoplasm], {'rprops': cyto_props}, prefix='Cytoplasm.',
        ))

    # compute nuclei gradient features
    feature_list.append(conditional(
        gradient_features_flag,
        compute_gradient_features,
        [im_label, im_nuclei], {'rprops': nuclei_props}, prefix='Nucleus.',
    ))

    # compute cytoplasm gradient features
    if im_cytoplasm is not None:
        feature_list.append(conditional(
            gradient_features_flag,
            compute_gradient_features,
            [cyto_mask, im_cytoplasm], {'rprops': cyto_props}, prefix='Cytoplasm.',
        ))

    # compute nuclei haralick features
    feature_list.append(conditional(
        haralick_features_flag,
        compute_haralick_features,
        [im_label, im_nuclei], {'num_levels': num_glcm_levels, 'rprops': nuclei_props},
        prefix='Nucleus.',
    ))

    # compute cytoplasm haralick features
    if im_cytoplasm is not None:
        feature_list.append(conditional(
            haralick_features_flag,
            compute_haralick_features,
            [cyto_mask, im_cytoplasm], {'num_levels': num_glcm_levels, 'rprops': cyto_props},
            prefix='Cytoplasm.',
        ))

    # Merge all features
    fdata = pd.concat(feature_list, axis=1)

    if return_nuclei_annotation:
        # Create nuclei segmentation with the generated regionprops
        nuclei_annot_list, selected_rows = cli_utils.create_tile_nuclei_annotations(
            im_nuclei_seg_mask, tile_info, format, nuclei_props)

        # Drop all rows which are not found in nuclei detection
        fdata = fdata[fdata.index.isin(selected_rows)]
        return fdata, nuclei_annot_list

    return fdata
