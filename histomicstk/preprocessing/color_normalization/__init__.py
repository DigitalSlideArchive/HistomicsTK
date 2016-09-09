"""
This package contains functions to correct non-uniform staining issues in
histopathology images.
"""

# make functions available at the package level using shadow imports
# since we mostly have one function per file
from .reinhard import reinhard
from .ReinhardSample import ReinhardSample

# list out things that are available for public use
__all__ = (

    # functions and classes of this package
    'reinhard',
    'ReinhardSample',
)
