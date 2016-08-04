import pandas as pd
from skimage.measure import regionprops

from .ComputeFSDFeatures import ComputeFSDFeatures
from .ComputeGradientFeatures import ComputeGradientFeatures
from .ComputeIntensityFeatures import ComputeIntensityFeatures
from .ComputeMorphometryFeatures import ComputeMorphometryFeatures

from histomicstk.segmentation import label as htk_label


def ExtractNuclearFeatures(im_label, im_nuclei, im_cytoplasm,
                           fsd_k=128, fsd_freq_bins=6, cyto_width=8):
    """
    Calculates features from a label image.

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

    fsd_k : int, optional
        Number of points for boundary resampling to calculate fourier
        descriptors. Default value = 128.

    fsd_freq_bins : int, optional
        Number of frequency bins for calculating FSDs. Default value = 6.

    cyto_width : scalar
        Estimated width of the ring-like neighborhood region around each
        nucleus to be considered as its cytoplasm. Default value = 8.

    Returns
    -------
    fdata : pandas.DataFrame
        A pandas data frame containing the features listed below for each
        object/label

    Notes
    -----
    List of features computed by this function

    Morphometry (size and shape) features of the nuclei
        See histomicstk.features.ComputeMorphometryFeatures for more details.
        Feature names prefixed by *Size.* or *Shape.*.

    Fourier shape descriptor features
        See `histomicstk.features.ComputeFSDFeatures` for more details.
        Feature names are prefixed by *FSD*.

    Intensity features for the nucleus and cytoplasm channels
        See `histomicstk.features.ComputeFSDFeatures` for more details.
        Feature names are prefixed by *Nucleus.Intensity.* for nucleus features
        and *Cytoplasm.Intensity.* for cytoplasm features.

    Gradient/edge features for the nucleus and cytoplasm channels
        See `histomicstk.features.ComputeGradientFeatures` for more details.
        Feature names are prefixed by *Nucleus.Gradient.* for nucleus features
        and *Cytoplasm.Gradient.* for cytoplasm features.

    See Also
    --------
    histomicstk.features.ComputeMorphometryFeatures,
    histomicstk.features.ComputeFSDFeatures,
    histomicstk.features.ComputeIntensityFeatures,
    histomicstk.features.ComputeGradientFeatures,
    """

    # get the number of objects in im_label
    regions = regionprops(im_label)

    # compute cytoplasm mask
    cyto_mask = htk_label.ComputeNeighborhoodMask(im_label,
                                                  neigh_width=cyto_width)

    # compute morphometry features
    fmorph = ComputeMorphometryFeatures(im_label, rprops=regions)

    # compute FSD features
    ffsd = ComputeFSDFeatures(im_label, fsd_k, fsd_freq_bins, cyto_width,
                              rprops=regions)

    # compute nucleus intensity features
    fint_nuclei = ComputeIntensityFeatures(im_label, im_nuclei, rprops=regions)
    fint_nuclei.columns = ['Nucleus.' + col for col in fint_nuclei.columns]

    # compute cytoplasm intensity features
    fint_cytoplasm = ComputeIntensityFeatures(cyto_mask, im_cytoplasm)
    fint_cytoplasm.columns = ['Cytoplasm.' + col
                              for col in fint_cytoplasm.columns]

    # compute nucleus gradient features
    fgrad_nuclei = ComputeGradientFeatures(im_label, im_nuclei, rprops=regions)
    fgrad_nuclei.columns = ['Nucleus.' + col for col in fgrad_nuclei.columns]

    # compute cytoplasm gradient features
    fgrad_cytoplasm = ComputeGradientFeatures(cyto_mask, im_cytoplasm)
    fgrad_cytoplasm.columns = ['Cytoplasm.' + col
                               for col in fgrad_cytoplasm.columns]

    # Merge all features
    fdata = pd.concat([
        fmorph,
        ffsd,
        fint_nuclei,
        fint_cytoplasm,
        fgrad_nuclei,
        fgrad_cytoplasm
    ], axis=1)

    return fdata
