"""
This package contains functions for segmenting a variety of objects/structures
(e.g. nuclei, tissue, cytoplasm) found in histopathology images.
"""

# make functions available at the package level using shadow imports
# since we mostly have one function per file
from histomicstk.utils.SimpleMask import SimpleMask
from .EmbedBounds import EmbedBounds
from .GraphColorSequential import GraphColorSequential
from .LabelRegionAdjacency import LabelRegionAdjacency
from .RegionAdjacencyLayer import RegionAdjacencyLayer

# import sub-packages to support nested calls
from . import label
from . import level_set
from . import nuclear

# list out things that are available for public use
__all__ = (

    # functions and classes of this package
    'EmbedBounds',
    'GraphColorSequential',
    'LabelRegionAdjacency',
    'RegionAdjacencyLayer',
    'SimpleMask',

    # sub-packages
    'label',
    'level_set',
    'nuclear',
)
