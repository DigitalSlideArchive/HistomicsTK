# make functions available at the package level using these shadow imports
# since we mostly have one function per file
from .rgb_to_od import rgb_to_od
from .od_to_rgb import od_to_rgb
from .rgb_lab import rgb_to_lab
from .lab_to_rgb import lab_to_rgb

# list out things that are available for public use
__all__ = (

    # functions and classes of this package
    'rgb_to_od',
    'od_to_rgb',
    'rgb_to_lab',
    'lab_to_rgb',
)
