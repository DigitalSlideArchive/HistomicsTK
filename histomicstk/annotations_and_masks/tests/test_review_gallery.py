import os
import shutil
import sys
import tempfile

import pytest
from pandas import read_csv

from histomicstk.annotations_and_masks.review_gallery import (
    _plot_rapid_review_vis, create_review_galleries,
    get_all_rois_from_folder_v2)

thisDir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(thisDir, '../../../tests'))
import tests.htk_test_utilities as utilities  # noqa
from tests.htk_test_utilities import getTestFilePath, girderClient  # noqa

# # for protyping
# from tests.htk_test_utilities import _connect_to_existing_local_dsa
# girderClient = _connect_to_existing_local_dsa()


class Cfg:
    def __init__(self):
        self.gc = None
        self.URL = None
        self.GTCodes_dict = None
        self.BASE_SAVEPATH = None
        self.SAVEPATHS = None
        self.combinedvis_savepath = None
        self.folderid = None
        self.post_folderid = None


cfg = Cfg()

# pytest runs tests in the order they appear in the module
@pytest.mark.usefixtures('girderClient')  # noqa
def test_prep(girderClient):  # noqa

    cfg.gc = girderClient
    cfg.URL = cfg.gc.urlBase.replace('api/v1/', '')

    # GT codes dict for parsing into label mask
    gtcodePath = getTestFilePath('sample_GTcodes.csv')
    cfg.GTCodes_dict = read_csv(gtcodePath)
    cfg.GTCodes_dict.index = cfg.GTCodes_dict.loc[:, 'group']
    cfg.GTCodes_dict = cfg.GTCodes_dict.to_dict(orient='index')

    # just a temp directory to save masks
    cfg.BASE_SAVEPATH = tempfile.mkdtemp()
    cfg.SAVEPATHS = {
        'contours': os.path.join(cfg.BASE_SAVEPATH, 'contours'),
        'rgb': os.path.join(cfg.BASE_SAVEPATH, 'rgbs'),
        'visualization': os.path.join(cfg.BASE_SAVEPATH, 'vis'),
    }
    for _, savepath in cfg.SAVEPATHS.items():
        os.mkdir(savepath)

    # where to save gallery
    cfg.combinedvis_savepath = os.path.join(cfg.BASE_SAVEPATH, 'combinedvis')
    os.mkdir(cfg.combinedvis_savepath)

    # get original item
    iteminfo = cfg.gc.get('/item', parameters={
        'text': "TCGA-A2-A0YE-01Z-00-DX1"})[0]

    # create the folder to parse its contents into galleries
    folderinfo = cfg.gc.post(
        '/folder', data={
            'parentId': iteminfo['folderId'],
            'name': 'test'
        })
    cfg.folderid = folderinfo['_id']

    # create girder folder to post resultant galleries
    post_folderinfo = cfg.gc.post(
        '/folder', data={
            'parentId': iteminfo['folderId'],
            'name': 'test-post'
        })
    cfg.post_folderid = post_folderinfo['_id']

    # copy the item multiple times to create dummy database
    for i in range(2):
        _ = cfg.gc.post(
            "/item/%s/copy" % iteminfo['_id'], data={
                'name': 'testSlide%dForRevGal' % i,
                'copyAnnotations': True,
                'folderId': cfg.folderid,
            })


class TestReviewGallery:
    """Test methods for getting review gallery."""

    def test_get_all_rois_from_folder_v2(self):
        """Test get_all_rois_from_folder_v2()."""
        if sys.version_info < (3, ):
            return
        # params for getting all rois for slide
        get_all_rois_kwargs = {
            'GTCodes_dict': cfg.GTCodes_dict,
            'save_directories': cfg.SAVEPATHS,
            'annotations_to_contours_kwargs': {
                'MPP': 5.0,
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
                'combinedvis_savepath': cfg.combinedvis_savepath,
                'zoomout': 1.5,

            },
        }

        # Get al rois to prep for gallery
        get_all_rois_from_folder_v2(
            gc=cfg.gc, folderid=cfg.folderid,
            get_all_rois_kwargs=get_all_rois_kwargs, monitor='test')

        # a couple of checks
        all_fovs = os.listdir(cfg.combinedvis_savepath)
        fovdict = dict()
        for fov in all_fovs:
            slide = fov.split('_')[0]
            if slide not in fovdict.keys():
                fovdict[slide] = []
            fovdict[slide].append(fov.split(slide)[1].split('.png')[0])

        assert set(fovdict.keys()) == {
            'testSlide0ForRevGal', 'testSlide1ForRevGal'}

        assert {
            j.split('_left-')[1] for j in fovdict['testSlide0ForRevGal']
            } == {
                '%d_top-%d_bottom-%d_right-%d'
                % (l, t, b, r) for (l, t, b, r) in [
                    (57584, 35788, 37425, 59421),
                    (58463, 38203, 39760, 60379),
                    (59181, 33473, 38043, 63712),
                ]
            }

    def test_create_review_galleries(self):
        """Test create_review_galleries()."""
        if sys.version_info < (3, ):
            return
        create_review_galleries_kwargs = {
            'tilepath_base': cfg.combinedvis_savepath,
            'upload_results': True,
            'gc': cfg.gc,
            'url': cfg.URL,
            'gallery_folderid': cfg.post_folderid,
            'gallery_savepath': None,
            'padding': 25,
            'tiles_per_row': 2,
            'tiles_per_column': 5,
            'nameprefix': 'test',
        }

        # create (+/- post) review gallery
        resps = create_review_galleries(**create_review_galleries_kwargs)

        assert len(resps) == 1
        assert {j['name'] for j in resps} == {'test_gallery-1'}

        # Now you can go to the histomicstk folder POST_FOLDERID
        # to check that the visualizations and gallery make sense!

        # cleanup
        shutil.rmtree(cfg.BASE_SAVEPATH)
