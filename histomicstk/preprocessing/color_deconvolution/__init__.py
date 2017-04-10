"""
This package contains implementation of methods to deconvolve or separate
the stains of histopathology images.
"""
# make functions available at the package level using shadow imports
# since we mostly have one function per file
from .color_convolution import color_convolution
from .complement_stain_matrix import complement_stain_matrix
from .sparse_color_deconvolution import sparse_color_deconvolution
from .separate_stains_macenko_pca import separate_stains_macenko_pca
from .separate_stains_xu_snmf import separate_stains_xu_snmf
from .rgb_separate_stains_macenko_pca import rgb_separate_stains_macenko_pca

# must be imported after ComplementStainMatrix
from .color_deconvolution import color_deconvolution

# list out things that are available for public use
__all__ = (

    # functions and classes of this package
    'color_convolution',
    'color_deconvolution',
    'complement_stain_matrix',
    'sparse_color_deconvolution',
    'separate_stains_macenko_pca',
    'separate_stains_xu_snmf',
    'rgb_separate_stains_macenko_pca',
)
