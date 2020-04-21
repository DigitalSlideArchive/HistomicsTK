import pytest
import os
import tempfile
import shutil
from imageio import imread
from pandas import read_csv
from histomicstk.annotations_and_masks.review_gallery import \
    get_all_rois_from_folder_v2, _plot_rapid_review_vis,\
    create_review_galleries

import sys
thisDir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(thisDir, '../../../tests'))
import tests.htk_test_utilities as utilities  # noqa
from tests.htk_test_utilities import girderClient, getTestFilePath  # noqa

# # for protyping
# from tests.htk_test_utilities import _connect_to_existing_local_dsa
# girderClient = _connect_to_existing_local_dsa()

global gc, folderid, URL, GTCodes_dict, BASE_SAVEPATH, SAVEPATHS, \
    combinedvis_savepath, folderid, post_folderid

# pytest runs tests in the order they appear in the module
@pytest.mark.usefixtures('girderClient')  # noqa
def test_prep(girderClient):  # noqa
    global gc, folderid, URL, GTCodes_dict, BASE_SAVEPATH, SAVEPATHS, \
        combinedvis_savepath, folderid, post_folderid

    gc = girderClient
    URL = 'http://localhost:8080/'

    # GT codes dict for parsing into label mask
    gtcodePath = getTestFilePath('sample_GTcodes.csv')
    GTCodes_dict = read_csv(gtcodePath)
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

    # get original item
    iteminfo = gc.get('/item', parameters={
        'text': "TCGA-A2-A0YE-01Z-00-DX1"})[0]

    # create the folder to parse its contents into galleries
    folderinfo = gc.post(
        '/folder', data={
            'parentId': iteminfo['folderId'],
            'name': 'test'
        })
    folderid = folderinfo['_id']

    # create girder folder to post resultant galleries
    post_folderinfo = gc.post(
        '/folder', data={
            'parentId': iteminfo['folderId'],
            'name': 'test-post'
        })
    post_folderid = post_folderinfo['_id']

    # copy the item multiple times to create dummy database
    for i in range(2):
        _ = gc.post(
            "/item/%s/copy" % iteminfo['_id'], data={
                'name': 'testSlide%dForRevGal' % i,
                'copyAnnotations': True,
                'folderId': folderid,
            })


class TestReviewGallery(object):
    """Test methods for getting review gallery."""

    def test_get_all_rois_from_folder_v2(self):
        """Test get_all_rois_from_folder_v2()."""
        # params for getting all rois for slide
        get_all_rois_kwargs = {
            'GTCodes_dict': GTCodes_dict,
            'save_directories': SAVEPATHS,
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
                'combinedvis_savepath': combinedvis_savepath,
                'zoomout': 1.5,

            },
        }

        # Get al rois to prep for gallery
        get_all_rois_from_folder_v2(
            gc=gc, folderid=folderid,
            get_all_rois_kwargs=get_all_rois_kwargs, monitor='test')

        # a couple of checks
        all_fovs = os.listdir(combinedvis_savepath)
        fovdict = dict()
        for fov in all_fovs:
            slide = fov.split('_')[0]
            if slide not in fovdict.keys():
                fovdict[slide] = []
            fovdict[slide].append(fov.split(slide)[1].split('.png')[0])

        assert set(fovdict.keys()) == {
            'testSlide0ForRevGal', 'testSlide1ForRevGal'}

        assert set([
                j.split('_left-')[1] for j in fovdict['testSlide0ForRevGal']
            ]) == {
                '%d_top-%d_bottom-%d_right-%d'
                % (l, t, b, r) for (l, t, b, r) in [
                      (57584, 35788, 37425, 59421),
                      (58463, 38203, 39760, 60379),
                      (59181, 33473, 38043, 63712),
                ]
            }

    def test_create_review_galleries(self):
        """Test create_review_galleries()."""
        create_review_galleries_kwargs = {
            'tilepath_base': combinedvis_savepath,
            'upload_results': True,
            'gc': gc,
            'url': URL,
            'gallery_folderid': post_folderid,
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
        shutil.rmtree(BASE_SAVEPATH)
