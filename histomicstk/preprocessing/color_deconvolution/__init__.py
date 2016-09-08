"""
This package contains implementation of methods to deconvolve or separate
the stains of histopathology images.
"""
# make functions available at the package level using shadow imports
# since we mostly have one function per file
from .ColorConvolution import ColorConvolution
from .ComplementStainMatrix import ComplementStainMatrix
from .SparseColorDeconvolution import SparseColorDeconvolution

# must be imported after ComplementStainMatrix
from .ColorDeconvolution import ColorDeconvolution

# list out things that are available for public use
__all__ = (

    # functions and classes of this package
    'ColorConvolution',
    'ColorDeconvolution',
    'ComplementStainMatrix',
    'SparseColorDeconvolution',
)
