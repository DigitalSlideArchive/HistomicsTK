"""
This package contains utility functions to convert images between different
color spaces.
"""
# make functions available at the package level using these shadow imports
# since we mostly have one function per file
from .rgb_to_od import rgb_to_od
from .od_to_rgb import od_to_rgb
from .rgb_to_sda import rgb_to_sda
from .sda_to_rgb import sda_to_rgb
from .rgb_to_lab import rgb_to_lab
from .lab_to_rgb import lab_to_rgb
from .rgb_to_hsi import rgb_to_hsi

from .lab_mean_std import lab_mean_std  # after rgb_to_lab

# list out things that are available for public use
__all__ = (

    # functions and classes of this package
    'lab_mean_std',
    'lab_to_rgb',
    'od_to_rgb',
    'sda_to_rgb',
    'rgb_to_hsi',
    'rgb_to_lab',
    'rgb_to_od',
    'rgb_to_sda',
)
