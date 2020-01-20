# -*- coding: utf-8 -*-
import unittest
from histomicstk.utils.girder_convenience_utils import connect_to_api

# %%===========================================================================
# Constants & prep work

APIURL = 'http://candygram.neurology.emory.edu:8080/api/v1/'
APIKEY = 'kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb'

# %%===========================================================================


class GirderConvenienceTest(unittest.TestCase):
    """Test utilities for interaction with girder."""

    def test_connect_to_api(self):
        """Test get_image_from_htk_response."""
        gc = connect_to_api(APIURL, apikey=APIKEY)
        self.assertEqual(gc.urlBase, APIURL)


# %%===========================================================================


if __name__ == '__main__':
    unittest.main()
