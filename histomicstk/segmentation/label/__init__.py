"""
This package contains functions for post-processing labeled segmentation
masks produced by segmentation algorithms.
"""

# make functions available at the package level using shadow imports
# since we mostly have one function per file
from .CompactLabel import CompactLabel
from .ComputeNeighborhoodMask import ComputeNeighborhoodMask
from .CondenseLabel import CondenseLabel
from .DeleteLabel import DeleteLabel
from .LabelPerimeter import LabelPerimeter
from .ShuffleLabel import ShuffleLabel
from .trace_boundary import trace_boundary

# must be imported after CondenseLabel
from .AreaOpenLabel import AreaOpenLabel
from .SplitLabel import SplitLabel
from .WidthOpenLabel import WidthOpenLabel

# must be imported after TraceBounds
from .TraceLabel import TraceLabel

# list out things that are available for public use
__all__ = (

    # functions and classes of this package
    'AreaOpenLabel',
    'CompactLabel',
    'ComputeNeighborhoodMask',
    'CondenseLabel',
    'DeleteLabel',
    'LabelPerimeter',
    'ShuffleLabel',
    'SplitLabel',
    'TraceBounds',
    'trace_boundary',
    'TraceLabel',
    'WidthOpenLabel',
)
