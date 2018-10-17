from girder.models.item import Item
from girder.models.folder import Folder
from girder.models.user import User
from tests import base


def setUpModule():
    base.enabledPlugins.append('HistomicsTK')
    base.startServer()


def tearDownModule():
    base.stopServer()


class ImageBrowseEndpointsTest(base.TestCase):
    def setUp(self):
        base.TestCase.setUp(self)
        self.admin = User().createUser('admin', 'password', 'first', 'last', 'admin@email.com',
                                       admin=True)
        self.folder = list(Folder().childFolders(self.admin, 'user', user=self.admin))[0]
        self.items = [
            Item().createItem('item_%i' % i, creator=self.admin, folder=self.folder)
            for i in range(10)
        ]
        for item in self.items:
            # make the item look like an image
            item['largeImage'] = {
                'fileId': 'deadbeef'
            }
            Item().save(item)
        self.nonimage = Item().createItem('non-image', creator=self.admin, folder=self.folder)

    def testGetNextImage(self):
        resp = self.request(path='/item/%s/next_image' % str(self.items[0]['_id']),
                            user=self.admin)
        self.assertStatusOk(resp)
        self.assertEqual(resp.json['_id'], str(self.items[1]['_id']))

        resp = self.request(path='/item/%s/next_image' % str(self.items[-1]['_id']),
                            user=self.admin)
        self.assertStatusOk(resp)
        self.assertEqual(resp.json['_id'], str(self.items[0]['_id']))

    def testGetPreviousImage(self):
        resp = self.request(path='/item/%s/previous_image' % str(self.items[0]['_id']),
                            user=self.admin)
        self.assertStatusOk(resp)
        self.assertEqual(resp.json['_id'], str(self.items[-1]['_id']))

        resp = self.request(path='/item/%s/previous_image' % str(self.items[-1]['_id']),
                            user=self.admin)
        self.assertStatusOk(resp)
        self.assertEqual(resp.json['_id'], str(self.items[-2]['_id']))

    def testGetNextImageException(self):
        resp = self.request(path='/item/%s/next_image' % str(self.nonimage['_id']),
                            user=self.admin)
        self.assertStatus(resp, 404)
