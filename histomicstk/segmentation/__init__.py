"""
This package contains functions for segmenting a variety of objects/structures
(e.g. nuclei, tissue, cytoplasm) found in histopathology images.
"""

# make functions available at the package level using shadow imports
# since we mostly have one function per file
from histomicstk.utils.SimpleMask import SimpleMask
from .embed_boundaries import embed_boundaries
from .map_color import map_color
from .rag import rag
from .rag_add_layer import rag_add_layer

# import sub-packages to support nested calls
from . import label
from . import level_set
from . import nuclear

# list out things that are available for public use
__all__ = (

    # functions and classes of this package
    'embed_boundaries',
    'map_color',
    'rag',
    'rag_add_layer',
    'SimpleMask',

    # sub-packages
    'label',
    'level_set',
    'nuclear',
)
