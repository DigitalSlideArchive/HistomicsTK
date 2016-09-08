"""
This package contains functions to computing a variety of image-based features
that quantify the appearance and/or morphology of an objects/regions in the
image. These are needed for classifying objects (e.g. nuclei) and
regions (e.g. tissues) found in histopathology images.
"""

# make functions available at the package level using shadow imports
# since we mostly have one function per file
from .ComputeFSDFeatures import ComputeFSDFeatures
from .ComputeGradientFeatures import ComputeGradientFeatures
from .ComputeHaralickFeatures import ComputeHaralickFeatures
from .ComputeIntensityFeatures import ComputeIntensityFeatures
from .ComputeMorphometryFeatures import ComputeMorphometryFeatures
from .graycomatrixext import graycomatrixext

from .ComputeNucleiFeatures import ComputeNucleiFeatures

# list out things that are available for public use
__all__ = (

    # functions and classes of this package
    'ComputeFSDFeatures',
    'ComputeGradientFeatures',
    'ComputeHaralickFeatures',
    'ComputeIntensityFeatures',
    'ComputeMorphometryFeatures',
    'ComputeNucleiFeatures',
    'graycomatrixext',
)
