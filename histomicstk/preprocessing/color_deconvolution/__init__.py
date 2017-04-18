"""
This package contains implementation of methods to deconvolve or separate
the stains of histopathology images.
"""
# make functions available at the package level using shadow imports
# since we mostly have one function per file
from . import stain_color_map as _stain_color_map
from .color_convolution import color_convolution
from .complement_stain_matrix import complement_stain_matrix
from .sparse_color_deconvolution import sparse_color_deconvolution

# must be imported after ComplementStainMatrix
from .color_deconvolution import color_deconvolution

#: A dictionary of names for reference stain vectors
stain_color_map = _stain_color_map.stain_color_map

# list out things that are available for public use
__all__ = (

    # functions, classes, and constants of this package
    'color_convolution',
    'color_deconvolution',
    'complement_stain_matrix',
    'sparse_color_deconvolution',
    'stain_color_map',
)
