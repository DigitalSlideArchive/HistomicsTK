# -*- coding: utf-8 -*-
import unittest
import os
import tempfile
import shutil

from histomicstk.utils.girder_convenience_utils import (
    connect_to_api, backup_annotation_jsons)

# %%===========================================================================
# Constants & prep work

APIURL = 'http://candygram.neurology.emory.edu:8080/api/v1/'
APIKEY = 'kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb'

SAMPLE_FOLDER_ID = "5d9246f6bd4404c6b1faaa89"
SAMPLE_SLIDE_ID = '5d586d57bd4404c6b1f28640'

# %%===========================================================================


class GirderConvenienceTest(unittest.TestCase):
    """Test utilities for interaction with girder."""

    def test_connect_to_api(self):
        """Test get_image_from_htk_response."""
        gc = connect_to_api(APIURL, apikey=APIKEY)
        self.assertEqual(gc.urlBase, APIURL)

    def backup_annotation_jsons(self):
        """Test get_image_from_htk_response."""
        gc = connect_to_api(APIURL, apikey=APIKEY)

        savepath = tempfile.mkdtemp()

        def dummy(annotations, local, printme=''):
            print(printme)

        backup_annotation_jsons(
            gc, folderid=SAMPLE_FOLDER_ID, local=savepath,
            callback=dummy, callback_kwargs={'printme': "callback works!"})

        self.assertGreater(len(os.listdir(savepath)), 0)

        # cleanup
        shutil.rmtree(savepath)

# %%===========================================================================


if __name__ == '__main__':
    unittest.main()
