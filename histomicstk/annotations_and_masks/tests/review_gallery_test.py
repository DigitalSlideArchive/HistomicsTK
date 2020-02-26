import os
import tempfile
import shutil
import unittest

from imageio import imread
from pandas import read_csv
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
        print("Test get_all_rois_from_folder_v2()")

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

        # a couple of checks
        all_fovs = os.listdir(combinedvis_savepath)
        fovdict = dict()
        for fov in all_fovs:
            slide = fov[:23]
            if slide not in fovdict.keys():
                fovdict[slide] = []
            fovdict[slide].append(fov.split(slide)[1].split('.png')[0])

        self.assertSetEqual(
            set(fovdict.keys()),
            {'TCGA-A1-A0SP-01Z-00-DX1', 'TCGA-A7-A0DA-01Z-00-DX1'})

        self.assertSetEqual(
            set(fovdict['TCGA-A1-A0SP-01Z-00-DX1']),
            {'_id-5e2a2d77ddda5f83986d135b_left-%d_top-%d_bottom-%d_right-%d'
                % (l, t, b, r) for (l, t, b, r) in [
                  (10124, 56533, 56789, 10380), (8076, 56021, 56277, 8332),
                  (8332, 54485, 54741, 8588), (8076, 57045, 57301, 8332),
                  (9356, 53973, 54229, 9612), (8076, 55509, 55765, 8332),
                  (7308, 55253, 55509, 7564), (9100, 55765, 56021, 9356),
                  (9356, 55765, 56021, 9612), (8332, 57045, 57301, 8588),
                  (8844, 57301, 57557, 9100)]}
        )

        # shape & value check
        imname = 'TCGA-A1-A0SP-01Z-00-DX1_id-5e2a2d77ddda5f83986d135b' \
            + '_left-10124_top-56533_bottom-56789_right-10380'
        cvis = imread(os.path.join(combinedvis_savepath, imname + '.png'))
        self.assertTupleEqual(cvis.shape, (322, 966, 3))

    def test_create_review_galleries(self):
        """Test create_review_galleries()."""
        print("Test create_review_galleries()")

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

        self.assertEqual(len(resps), 3)
        self.assertSetEqual(
            {j['name'] for j in resps},
            {'gallery-1', 'gallery-2', 'gallery-3'})

        # Now you can go to the histomicstk folder POST_FOLDERID
        # to check that the visualizations and galler make sense!

# %%===========================================================================


def suite():
    """Run chained unit tests in desired order.

    See: https://stackoverflow.com/questions/5387299/...
         ... python-unittest-testcase-execution-order
    """
    suite = unittest.TestSuite()
    suite.addTest(ReviewGallery('test_get_all_rois_from_folder_v2'))
    suite.addTest(ReviewGallery('test_create_review_galleries'))

    return suite

# %%===========================================================================


if __name__ == '__main__':
    runner = unittest.TextTestRunner(failfast=True)
    runner.run(suite())

    # cleanup
    shutil.rmtree(BASE_SAVEPATH)
