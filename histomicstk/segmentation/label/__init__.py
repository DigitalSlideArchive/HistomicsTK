# make functions available at the package level using shadow imports
# since we mostly have one function per file
from .CondenseLabel import CondenseLabel
from .CompactLabel import CompactLabel
from .DeleteLabel import DeleteLabel
from .LabelPerimeter import LabelPerimeter
from .ShuffleLabel import ShuffleLabel
from .TraceBounds import TraceBounds

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
    'CondenseLabel',
    'DeleteLabel'
    'LabelPerimeter',
    'ShuffleLabel',
    'SplitLabel',
    'TraceBounds',
    'TraceLabel',
    'WidthOpenLabel',
)
