"""
This package contains functions for post-processing labeled segmentation
masks produced by segmentation algorithms.
"""

# make functions available at the package level using shadow imports
# since we mostly have one function per file
from .compact import compact
from .dilate_xor import dilate_xor
from .condense import condense
from .delete import delete
from .delete_border import delete_border
from .perimeter import perimeter
from .shuffle import shuffle
from .trace_object_boundaries import trace_object_boundaries

# must be imported after CondenseLabel
from .area_open import area_open
from .split import split
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
)
