"""
This package contains functions to enhance edges in images.
"""
# make functions available at the package level using shadow imports
# since we mostly have one function per file
from .gaussian_grad import gaussian_grad

# list out things that are available for public use
__all__ = (

    # functions and classes of this package
    'gaussian_grad',
)
