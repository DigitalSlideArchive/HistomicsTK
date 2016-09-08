"""
This package contains utility functions that are widely used by functions in
all other sub-packages of histomicstk
"""

# make functions available at the package level using shadow imports
# since we mostly have one function per file
from .Del2 import Del2
from .Eigenvalues import Eigenvalues
from .GradientDiffusion import GradientDiffusion
from .Hessian import Hessian
from .MergeColinear import MergeColinear
from .PoissonMixture import PoissonMixture
from .SimpleMask import SimpleMask
from .Sample import Sample  # must import after SimpleMask


# list out things that are available for public use
__all__ = (

    # functions and classes of this package
    'Del2',
    'Eigenvalues',
    'GradientDiffusion',
    'Hessian',
    'MergeColinear',
    'PoissonMixture',
    'Sample',
    'SimpleMask',
)
