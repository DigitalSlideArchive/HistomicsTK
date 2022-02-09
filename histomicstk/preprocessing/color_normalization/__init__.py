"""Functions to correct non-uniform staining issues."""

# make functions available at the package level using shadow imports
# since we mostly have one function per file
from .background_intensity import background_intensity
from .deconvolution_based_normalization import \
    deconvolution_based_normalization
from .reinhard import reinhard
from .reinhard_stats import reinhard_stats

# list out things that are available for public use
__all__ = (

    # functions and classes of this package
    'background_intensity',
    'reinhard',
    'reinhard_stats',
    'deconvolution_based_normalization',
)
