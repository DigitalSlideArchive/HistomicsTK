# -*- coding: utf-8 -*-

import tests.htk_test_utilities as utilities
from tests.htk_test_utilities import girderClient  # noqa
from histomicstk.utils.girder_convenience_utils import connect_to_api, \
    update_permissions_for_annotation, update_styles_for_annotations_in_slide

# %%===========================================================================


class GirderConvenienceTest(object):
    """Test utilities for interaction with girder."""

    @pytest.mark.usefixtures('girderClient')  # noqa
    def test_update_permissions_for_annotation(self, girderClient):  # noqa
        annid = "5e2a2d77ddda5f83986d135d"
        resp = update_permissions_for_annotation(
            gc=girderClient, annotation_id=annid,
            users_to_add=[
                {'login': 'kheffah',
                    'level': 2, 'id': '59bc677892ca9a0017c2e855'},
                {'login': 'testing',
                    'level': 0, 'id': '5d588370bd4404c6b1f28933'},
            ],
            replace_original_users=True
        )
        # self.assertListEqual(
        #     resp['access']['users'],
        #     [{'flags': [], 'id': '59bc677892ca9a0017c2e855', 'level': 2},
        #      {'flags': [], 'id': '5d588370bd4404c6b1f28933', 'level': 0}]
        # )

    # def test_update_styles_for_annotations_in_slide(self):
    #
    #     resps = update_styles_for_annotations_in_slide(
    #         gc=GC, slide_id='5e2a2d77ddda5f83986d135b',
    #         changes={
    #             'fov_discordant': {
    #                 'group': 'fov_discordant',
    #                 'lineColor': 'rgb(131,181,255)',
    #                 'fillColor': 'rgba(131,181,255,0.3)',
    #             },
    #         },
    #         monitorPrefix='test',
    #     )
    #
    #     unique_groups = set()
    #     for ann in resps:
    #         if ann is not None:
    #             unique_groups = unique_groups.union(set(ann['groups']))
    #     self.assertSetEqual(
    #         unique_groups,
    #         {'fibroblast', 'fov_discordant', 'lymphocyte',
    #          'tumor', 'unlabeled'},
    #     )


# %%===========================================================================


# if __name__ == '__main__':
#     unittest.main()