import numpy as np
import pandas as pd
from skimage.measure import regionprops

from .ComputeFSDFeatures import ComputeFSDFeatures
from .ComputeGradientFeatures import ComputeGradientFeatures
from .ComputeIntensityFeatures import ComputeIntensityFeatures
from .ComputeMorphometryFeatures import ComputeMorphometryFeatures

from histomicstk.segmentation import label as htk_label


def FeatureExtraction(Label, In, Ic, K=128, Fs=6, Delta=8):
    """
    Calculates features from a label image.

    Parameters
    ----------
    Label : array_like
        A M x N label image.
    In : array_like
        A M x N intensity image for Nuclei.
    Ic : array_like
        A M x N intensity image for Cytoplasms.
    K : Number of points for boundary resampling to calculate fourier
        descriptors. Default value = 128.
    Fs : Number of frequency bins for calculating FSDs. Default value = 6.
    Delta : scalar, used to dilate nuclei and define cytoplasm region.
            Default value = 8.

    Returns
    -------
    df : 2-dimensional labeled data structure, float64
        Pandas data frame.

    Notes
    -----
    The following features are computed:

    - `Morphometry features`:
        - CentroidsX,
        - CentroidsY,
        - Area,
        - Perimeter,
        - MajorAxisLength,
        - MinorAxisLength,
        - MajorMinorAxisRatio,
        - MajorAxisCoordsX,
        - MajorAxisCoordsY,
        - Eccentricity,
        - Circularity,
        - Extent,
        - Solidity

    - `Fourier shape descriptors`:
        - FSD1-FSD6

    - Intensity features for hematoxylin and cytoplasm channels:
        - MinIntensity, MaxIntensity,
        - MeanIntensity, StdIntensity,
        - MeanMedianDifferenceIntensity,
        - Entropy, Energy, Skewness and Kurtosis

    - Gradient/edge features for hematoxylin and cytoplasm channels:
        - MeanGradMag, StdGradMag, SkewnessGradMag, KurtosisGradMag,
        - EntropyGradMag, EnergyGradMag,
        - SumCanny, MeanCanny

    References
    ----------
    .. [1] D. Zhang et al. "A comparative study on shape retrieval using
       Fourier descriptors with different shape signatures," In Proc.
       ICIMADE01, 2001.
    .. [2] Daniel Zwillinger and Stephen Kokoska. "CRC standard probability
       and statistics tables and formulae," Crc Press, 1999.
    """

    # get the number of objects in Label
    regions = regionprops(Label)

    # initialize panda dataframe
    df = pd.DataFrame()

    fmorph = ComputeMorphometryFeatures(Label, rprops=regions)
    df = pd.concat([df, fmorph], axis=1)

    ffsds = ComputeFSDFeatures(Label, K, Fs, Delta, rprops=regions)
    df = pd.concat([df, ffsds], axis=1)

    fint_nuclei = ComputeIntensityFeatures(Label, In, rprops=regions)
    fint_nuclei.columns = ['Nucleus.' + col for col in fint_nuclei.columns]
    df = pd.concat([df, fint_nuclei], axis=1)

    cyto_mask = htk_label.ComputeNeighborhoodMask(Label, neigh_width=Delta)
    fint_cytoplasm = ComputeIntensityFeatures(cyto_mask, Ic)
    fint_cytoplasm.columns = ['Cytoplasm.' + col
                              for col in fint_cytoplasm.columns]
    df = pd.concat([df, fint_cytoplasm], axis=1)

    fgrad_nuclei = ComputeGradientFeatures(Label, In, rprops=regions)
    fgrad_nuclei.columns = ['Nucleus.' + col for col in fgrad_nuclei.columns]
    df = pd.concat([df, fgrad_nuclei], axis=1)

    fgrad_cytoplasm = ComputeGradientFeatures(cyto_mask, Ic)
    fgrad_cytoplasm.columns = ['Cytoplasm.' + col
                               for col in fgrad_cytoplasm.columns]
    df = pd.concat([df, fgrad_cytoplasm], axis=1)

    return df
