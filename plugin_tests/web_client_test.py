import os

from tests import web_client_test

setUpModule = web_client_test.setUpModule
tearDownModule = web_client_test.tearDownModule

TEST_FILE = os.path.join(
    os.environ['GIRDER_TEST_DATA_PREFIX'],
    'plugins', 'HistomicsTK',
    'sample_svs_image.TCGA-DU-6399-01A-01-TS1.e8eb65de-d63e-42db-af6f-14fefbbdf7bd.svs'  # noqa
)


class WebClientTestCase(web_client_test.WebClientTestCase):

    def setUp(self):
        super(WebClientTestCase, self).setUp()

        user = self.model('user').createUser(
            login='admin',
            password='password',
            email='admin@email.com',
            firstName='Admin',
            lastName='Admin',
            admin=True
        )

        publicFolder = self.model('folder').childFolders(
            user, 'user', filters={'name': 'Public'}
        ).next()

        with open(TEST_FILE) as f:
            file = self.uploadFile('image', f.read(), user, publicFolder)

        item = self.model('item').load(file['itemId'], force=True)
        self.model('image_item', 'large_image').createImageItem(
            item, file, user=user, createJob=False
        )
