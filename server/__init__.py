import os

from girder.api.rest import Resource, loadmodel, getApiUrl
from girder.api import access
from girder.api.describe import Description
from girder.constants import AccessType


class DeepakTest(Resource):
    def __init__(self):
        self.resourceName = 'deepak_test'

        self.route('POST', ('analysis',), self.doAnalysis)

    @access.user
    @loadmodel(map={'itemId': 'item'}, model='item', level=AccessType.READ)
    @loadmodel(map={'folderId': 'folder'}, model='folder', level=AccessType.WRITE)
    def doAnalysis(self, item, folder, params):
        with open(os.path.join(os.path.dirname(__file__), 'script.py')) as f:
            codeToRun = f.read()

        jobModel = self.model('job', 'jobs')
        user = self.getCurrentUser()

        job = jobModel.createJob(
            title='test', type='deepak_test', handler='romanesco_handler',
            user=user)
        jobToken = jobModel.createJobToken(job)
        token = self.getCurrentToken()['_id']

        kwargs = {
            'task': {
                'name': 'Deepak test',
                'mode': 'python',
                'script': codeToRun,
                'inputs': [{
                    'id': 'imageFile',
                    'type': 'string',
                    'format': 'string',
                    'target': 'filepath'
                }],
                'outputs': [{
                    'id': 'resultImage',
                    'type': 'string',
                    'format': 'string'
                }]
            },
            'inputs': {
                'imageFile': {
                    "mode": "girder",
                    "id": str(item['_id']),
                    "name": item['name'],
                    "host": 'localhost',
                    "format": "string",
                    "type": "string",
                    'port': 8080,
                    'token': token,
                    'resource_type': 'item'
                }
            },
            'outputs': {
                'resultImage': {
                    "mode": "girder",
                    "parent_id": str(folder['_id']),
                    "host": 'localhost',
                    "format": "string",
                    "type": "string",
                    'port': 8080,
                    'token': token,
                    'parent_type': 'folder'
                }
            },
            'jobInfo': {
                'method': 'PUT',
                'url': '/'.join((getApiUrl(), 'job', str(job['_id']))),
                'headers': {'Girder-Token': jobToken['_id']},
                'logPrint': True
            },
            'validate': False,
            'auto_convert': False,
            'cleanup': True
        }
        job['kwargs'] = kwargs
        job = jobModel.save(job)
        jobModel.scheduleJob(job)

        return jobModel.filter(job, user)
    doAnalysis.description = (
        Description('Run color deconvolution on an image.')
        .param('itemId', 'ID of the item containing the image.')
        .param('folderId', 'ID of the output folder.'))

def load(info):
    info['apiRoot'].deepak_test = DeepakTest()
