# -*- coding: utf-8 -*-
import unittest
import os
import tempfile
import shutil

from histomicstk.utils.girder_convenience_utils import connect_to_api
from histomicstk.annotations_and_masks.annotation_database_parser import (
    dump_annotations_locally, )

# %%===========================================================================
# Constants & prep work

APIURL = 'http://candygram.neurology.emory.edu:8080/api/v1/'
APIKEY = 'kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb'

SAMPLE_FOLDER_ID = "5d9246f6bd4404c6b1faaa89"
SAMPLE_SLIDE_ID = '5d586d57bd4404c6b1f28640'

# %%===========================================================================


class DatabaseParserTest(unittest.TestCase):
    """Test girder database parser."""

    def dump_annotations_locally_test(self):
        """Test get_image_from_htk_response."""
        gc = connect_to_api(APIURL, apikey=APIKEY)

        savepath = tempfile.mkdtemp()

        def dummy(annotations, local, printme=''):
            print(printme)

        dump_annotations_locally(
            gc, folderid=SAMPLE_FOLDER_ID, local=savepath,
            callback=dummy, callback_kwargs={'printme': "callback works!"})

        self.assertGreater(len(os.listdir(savepath)), 0)

        # cleanup
        shutil.rmtree(savepath)

# %%===========================================================================


if __name__ == '__main__':
    unittest.main()
