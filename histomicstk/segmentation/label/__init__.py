"""
This package contains functions for post-processing labeled segmentation
masks produced by segmentation algorithms.
"""

# must be imported after CondenseLabel
from .area_open import area_open
# make functions available at the package level using shadow imports
# since we mostly have one function per file
from .compact import compact
from .condense import condense
from .delete import delete
from .delete_border import delete_border
from .delete_overlap import delete_overlap
from .dilate_xor import dilate_xor
from .perimeter import perimeter
from .shuffle import shuffle
from .split import split
from .trace_object_boundaries import trace_object_boundaries
from .width_open import width_open

# list out things that are available for public use
__all__ = (

    # functions and classes of this package
    'area_open',
    'compact',
    'dilate_xor',
    'condense',
    'delete',
    'delete_border',
    'perimeter',
    'shuffle',
    'split',
    'trace_object_boundaries',
    'width_open',
    'delete_overlap'
)
