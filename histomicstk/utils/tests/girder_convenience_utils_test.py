# -*- coding: utf-8 -*-
import unittest
# import os
# import girder_client

from histomicstk.utils.girder_convenience_utils import (
    connect_to_api, get_items, backup_annotation_jsons)

# %%===========================================================================
# Constants & prep work

APIURL = 'http://candygram5.neurology.emory.edu:8080/api/v1/'
SAMPLE_FOLDER_ID = "59a5c4e692ca9a00174d77d7"
SAMPLE_SLIDE_ID = '5d586d57bd4404c6b1f28640'

# %%===========================================================================


class GirderConvenienceTest(unittest.TestCase):
    """Test utilities for interaction with girder."""

    def test_connect_to_api(self):
        """Test get_image_from_htk_response."""
        pass

# %%===========================================================================


# if __name__ =
# = '__main__':
#     unittest.main()
