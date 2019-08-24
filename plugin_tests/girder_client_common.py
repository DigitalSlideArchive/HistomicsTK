import girder_client
import os

from girder import config
from girder.models.user import User

from tests import base


TEST_DATA_DIR = os.path.join(os.environ['GIRDER_TEST_DATA_PREFIX'], 'plugins/HistomicsTK')

GTCODE_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_files', 'sample_GTcodes.csv')


os.environ['GIRDER_PORT'] = os.environ.get('GIRDER_TEST_PORT', '20200')
config.loadConfig()  # Must reload config to pickup correct port


class GirderClientTestCase(base.TestCase):
    def setUp(self):
        base.TestCase.setUp(self)
        admin = {
            'email': 'admin@email.com',
            'login': 'adminlogin',
            'firstName': 'Admin',
            'lastName': 'Last',
            'password': 'adminpassword',
            'admin': True
        }
        self.admin = User().createUser(**admin)
        self.gc = girder_client.GirderClient(
            apiUrl='http://127.0.0.1:%d/api/v1' % (int(os.environ['GIRDER_PORT'])))
        self.gc.authenticate(username='adminlogin', password='adminpassword')
        self.publicFolder = self.gc.get(
            'resource/lookup', parameters={'path': 'user/%s/Public' % self.admin['login']})
        wsi_path = os.path.join(
            TEST_DATA_DIR,
            'TCGA-A2-A0YE-01Z-00-DX1.8A2E3094-5755-42BC-969D-7F0A2ECA0F39.svs'
        )
        self.wsiFile = self.gc.uploadFile(
            self.publicFolder['_id'], stream=open(wsi_path, 'rb'),
            name=os.path.basename(wsi_path), size=os.stat(wsi_path).st_size,
            parentType='folder')
        annotations_path = wsi_path + '_annotations.json'
        self.gc.post(
            'annotation/item/%s' % self.wsiFile['itemId'],
            data=open(annotations_path, 'rb').read(),
            headers={'Content-Type': 'application/json'})
