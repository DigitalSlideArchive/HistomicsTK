"""
This package contains utility functions that are widely used by functions in
all other sub-packages of histomicstk
"""

from . import general_utils, girder_convenience_utils
# make functions available at the package level using shadow imports
# since we mostly have one function per file
from .compute_tile_foreground_fraction import compute_tile_foreground_fraction
from .convert_image_to_matrix import convert_image_to_matrix
from .convert_matrix_to_image import convert_matrix_to_image
from .del2 import del2
from .eigen import eigen
from .exclude_nonfinite import exclude_nonfinite
from .fit_poisson_mixture import fit_poisson_mixture
from .gradient_diffusion import gradient_diffusion
from .hessian import hessian
from .merge_colinear import merge_colinear
from .sample_pixels import sample_pixels  # must import after SimpleMask
from .simple_mask import simple_mask

# list out things that are available for public use
__all__ = (

    # functions and classes of this package
    'compute_tile_foreground_fraction',
    'convert_matrix_to_image',
    'convert_image_to_matrix',
    'del2',
    'eigen',
    'exclude_nonfinite',
    'gradient_diffusion',
    'hessian',
    'merge_colinear',
    'fit_poisson_mixture',
    'sample_pixels',
    'simple_mask',
    'general_utils',
    'girder_convenience_utils',
)
