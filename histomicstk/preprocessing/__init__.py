"""
This package contains functions to pre-process histopathology images.
"""
# import sub-packages to support nested calls
from . import color_conversion
from . import color_deconvolution
from . import color_normalization

# list out things that are available for public use
__all__ = (
    'color_conversion',
    'color_deconvolution',
    'color_normalization',
)
