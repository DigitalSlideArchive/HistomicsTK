"""
This package contains functions for enhancing different kinds of structures
(e.g. edges/membrane, blobs/nuclei, vessels) in images.
"""
# import sub-packages to support nested calls
from . import edge, shape

# list out things that are available for public use
__all__ = (
    # sub-packages
    'edge',
    'shape',
)
