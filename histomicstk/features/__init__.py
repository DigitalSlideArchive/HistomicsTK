"""
This package contains functions to computing a variety of image-based features
that quantify the appearance and/or morphology of an objects/regions in the
image. These are needed for classifying objects (e.g. nuclei) and
regions (e.g. tissues) found in histopathology images.
"""

# make functions available at the package level using shadow imports
# since we mostly have one function per file
from .compute_fsd_features import compute_fsd_features
from .compute_global_cell_graph_features import \
    compute_global_cell_graph_features
from .compute_gradient_features import compute_gradient_features
from .compute_haralick_features import compute_haralick_features
from .compute_intensity_features import compute_intensity_features
from .compute_morphometry_features import compute_morphometry_features
from .compute_nuclei_features import compute_nuclei_features
from .graycomatrixext import graycomatrixext

# list out things that are available for public use
__all__ = (

    # functions and classes of this package
    'compute_fsd_features',
    'compute_global_cell_graph_features',
    'compute_gradient_features',
    'compute_haralick_features',
    'compute_intensity_features',
    'compute_morphometry_features',
    'compute_nuclei_features',
    'graycomatrixext',
)
