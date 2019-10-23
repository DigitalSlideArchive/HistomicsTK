"""
This package contains implementation of methods to deconvolve or separate.

... the stains of histopathology images.
"""
# make functions available at the package level using shadow imports
# since we mostly have one function per file
from . import stain_color_map as _stain_color_map
from .color_convolution import color_convolution
from .complement_stain_matrix import complement_stain_matrix
from .find_stain_index import find_stain_index
from .separate_stains_macenko_pca import separate_stains_macenko_pca
from .separate_stains_xu_snmf import separate_stains_xu_snmf
from .rgb_separate_stains_macenko_pca import rgb_separate_stains_macenko_pca
from .rgb_separate_stains_xu_snmf import rgb_separate_stains_xu_snmf

# must be imported after ComplementStainMatrix
from .color_deconvolution import color_deconvolution
from .color_deconvolution import stain_unmixing_routine
from .color_deconvolution import color_deconvolution_routine
from .color_deconvolution import _reorder_stains

#: A dictionary of names for reference stain vectors
stain_color_map = _stain_color_map.stain_color_map

# list out things that are available for public use
__all__ = (

    # functions, classes, and constants of this package
    'color_convolution',
    'color_deconvolution',
    'stain_unmixing_routine',
    'color_deconvolution_routine',
    'complement_stain_matrix',
    'find_stain_index',
    'separate_stains_macenko_pca',
    'separate_stains_xu_snmf',
    'rgb_separate_stains_macenko_pca',
    'rgb_separate_stains_xu_snmf',
    'stain_color_map',
    '_reorder_stains',
)
