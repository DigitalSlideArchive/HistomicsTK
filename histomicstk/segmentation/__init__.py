"""
This package contains functions for segmenting a variety of objects/structures
(e.g. nuclei, tissue, cytoplasm) found in histopathology images.
"""

# make functions available at the package level using shadow imports
# since we mostly have one function per file
from histomicstk.utils.simple_mask import simple_mask

# import sub-packages to support nested calls
from . import label, level_set, nuclear, positive_pixel_count
from .embed_boundaries import embed_boundaries
from .rag import rag
from .rag_add_layer import rag_add_layer
from .rag_color import rag_color

# list out things that are available for public use
__all__ = (

    # functions and classes of this package
    'embed_boundaries',
    'rag_color',
    'rag',
    'rag_add_layer',
    'simple_mask',

    # sub-packages
    'label',
    'level_set',
    'nuclear',
    'positive_pixel_count',
)
