import unittest

import os
import tempfile
import shutil
import unittest

import matplotlib.pylab as plt
from imageio import imread
from pandas import read_csv
import numpy as np
import girder_client
from histomicstk.annotations_and_masks.review_gallery import \
    get_all_rois_from_folder_v2, _plot_rapid_review_vis,\
    create_review_galleries

# %%===========================================================================
# Constants & prep work

URL = 'http://candygram.neurology.emory.edu:8080/'
APIURL = URL + 'api/v1/'
SAMPLE_FOLDER_ID = '5e2a2da8ddda5f83986d18a2'

POST_FOLDERID = "5e3ce440ddda5f839875b33e"

gc = girder_client.GirderClient(apiUrl=APIURL)
gc.authenticate(interactive=True)  # need this to post!
# gc.authenticate(apiKey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')

# GT codes dict for parsing into label mask
GTCODE_PATH = os.path.join(
    '/home/mtageld/Desktop/HistomicsTK/histomicstk/annotations_and_masks/',
    'tests/test_files', 'sample_GTcodes_v2.csv')
GTCodes_dict = read_csv(GTCODE_PATH)
GTCodes_dict.index = GTCodes_dict.loc[:, 'group']
GTCodes_dict = GTCodes_dict.to_dict(orient='index')

# just a temp directory to save masks
BASE_SAVEPATH = tempfile.mkdtemp()
SAVEPATHS = {
    'contours': os.path.join(BASE_SAVEPATH, 'contours'),
    'rgb': os.path.join(BASE_SAVEPATH, 'rgbs'),
    'visualization': os.path.join(BASE_SAVEPATH, 'vis'),
}
for _, savepath in SAVEPATHS.items():
    os.mkdir(savepath)

# where to save gallery
combinedvis_savepath = os.path.join(BASE_SAVEPATH, 'combinedvis')
os.mkdir(combinedvis_savepath)

# %%===========================================================================


class ReviewGallery(unittest.TestCase):
    """Test methods for getting review gallery."""

    def test_get_all_rois_from_folder_v2(self):
        """Test get_all_rois_from_folder_v2()."""

        # params for getting all rois for slide
        get_all_rois_kwargs = {
            'GTCodes_dict': GTCodes_dict,
            'save_directories': SAVEPATHS,
            'annotations_to_contours_kwargs': {
                'MPP': 0.2,
                'linewidth': 0.2,
                'get_rgb': True,
                'get_visualization': True,
            },
            'verbose': False,
            'get_mask': False,
            # we use this callback so that we have results compatible
            # of being used as input for create_review_galleries()
            'callback': _plot_rapid_review_vis,
            'callback_kwargs': {
                'combinedvis_savepath': combinedvis_savepath,
                'zoomout': 4,

            },
        }

        # Get al rois to prep for gallery
        get_all_rois_from_folder_v2(
            gc=gc, folderid=SAMPLE_FOLDER_ID,
            get_all_rois_kwargs=get_all_rois_kwargs, monitor='test')

    def test_create_review_galleries(self):
        """Test create_review_galleries()."""

        create_review_galleries_kwargs = {
            'tilepath_base': combinedvis_savepath,
            'upload_results': True,
            'gc': gc,
            'url': URL,
            'gallery_folderid': POST_FOLDERID,
            'gallery_savepath': None,
            'padding': 25,
            'tiles_per_row': 2,
            'tiles_per_column': 5,
        }

        # create (+/- post) review gallery
        resps = create_review_galleries(**create_review_galleries_kwargs)

# %%===========================================================================

# def suite():
#     """Run chained unit tests in desired order.
#
#     See: https://stackoverflow.com/questions/5387299/...
#          ... python-unittest-testcase-execution-order
#     """
#     suite = unittest.TestSuite()
#     suite.addTest(TissueDetectionTest('test_get_tissue_mask'))
#     suite.addTest(
#         TissueDetectionTest('test_get_tissue_boundary_annotation_documents'))
#     return suite
#
# # %%===========================================================================
#
#
# if __name__ == '__main__':
#     runner = unittest.TextTestRunner(failfast=True)
#     runner.run(suite())
#     # cleanup
#     shutil.rmtree(savepath)