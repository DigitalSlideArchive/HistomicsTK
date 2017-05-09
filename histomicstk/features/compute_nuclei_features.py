import pandas as pd
from skimage.measure import regionprops

from .compute_fsd_features import compute_fsd_features
from .compute_gradient_features import compute_gradient_features
from .compute_haralick_features import compute_haralick_features
from .compute_intensity_features import compute_intensity_features
from .compute_morphometry_features import compute_morphometry_features

from histomicstk.segmentation import label as htk_label


def compute_nuclei_features(im_label, im_nuclei, im_cytoplasm=None,
                            fsd_bnd_pts=128, fsd_freq_bins=6, cyto_width=8,
                            num_glcm_levels=32,
                            morphometry_features_flag=True,
                            fsd_features_flag=True,
                            intensity_features_flag=True,
                            gradient_features_flag=True,
                            haralick_features_flag=True
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

    Returns
    -------
    fdata : pandas.DataFrame
        A pandas data frame containing the features listed below for each
        object/label

    Notes
    -----
    List of features computed by this function

    Morphometry (size and shape) features of the nuclei
        See histomicstk.features.compute_morphometry_features for more details.
        Feature names prefixed by *Size.* or *Shape.*.

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

    feature_list = []

    # get the number of objects in im_label
    nuclei_props = regionprops(im_label)

    # compute cytoplasm mask
    if im_cytoplasm is not None:

        cyto_mask = htk_label.dilate_xor(im_label, neigh_width=cyto_width)

        cytoplasm_props = regionprops(cyto_mask)

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
                                                    rprops=cytoplasm_props)
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
                                                    rprops=cytoplasm_props)
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
            rprops=cytoplasm_props
        )

        fharalick_cytoplasm.columns = ['Cytoplasm.' + col
                                       for col in fharalick_cytoplasm.columns]

        feature_list.append(fharalick_cytoplasm)

    # Merge all features
    fdata = pd.concat(feature_list, axis=1)

    return fdata
