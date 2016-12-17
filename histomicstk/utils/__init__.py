"""
This package contains utility functions that are widely used by functions in
all other sub-packages of histomicstk
"""

# make functions available at the package level using shadow imports
# since we mostly have one function per file
from .del2 import del2
from .eigen import eigen
from .gradient_diffusion import gradient_diffusion
from .hessian import hessian
from .merge_colinear import merge_colinear
from .fit_poisson_mixture import fit_poisson_mixture
from .simple_mask import simple_mask
from .sample_pixels import sample_pixels  # must import after SimpleMask

# list out things that are available for public use
__all__ = (

    # functions and classes of this package
    'del2',
    'eigen',
    'gradient_diffusion',
    'hessian',
    'merge_colinear',
    'fit_poisson_mixture',
    'sample_pixels',
    'simple_mask',
)
