# import sub-packages to support nested calls
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _importlib_version

from . import segmentation  # must be imported before features
from . import utils  # must be imported before other packages
from . import (annotations_and_masks, features, filters, preprocessing,
               saliency, workflows)

try:
    __version__ = _importlib_version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass


# list out things that are available for public use
__all__ = (

    # sub-packages
    'features',
    'filters',
    'preprocessing',
    'segmentation',
    'utils',
    'annotations_and_masks',
    'saliency',
    'workflows',
)
