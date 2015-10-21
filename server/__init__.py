import os

from girder.api.rest import Resource, loadmodel, getApiUrl
from girder.api import access
from girder.api.describe import Description
from girder.constants import AccessType

class ColorDeconvolution(Resource):
    def __init__(self):
        self.resourceName = 'ColorDeconvolution'
        self.route('POST', ('analysis',), self.doAnalysis)

    @access.user
    @loadmodel(map={'itemId': 'item'}, model='item', level=AccessType.READ)
    @loadmodel(map={'folderId': 'folder'}, model='folder', level=AccessType.WRITE)
    def doAnalysis(self, item, folder, params):
        with open(os.path.join(os.path.dirname(__file__), 'script.py')) as f:
            codeToRun = f.read()

        jobModel = self.model('job', 'jobs')
        user = self.getCurrentUser()

        job = jobModel.createJob(title='ColorDeconvolution',
                                 type='ColorDeconvolution',
                                 handler='romanesco_handler',
                                 user=user)
        jobToken = jobModel.createJobToken(job)
        token = self.getCurrentToken()['_id']

        kwargs = {
            'task': {
                'name': 'ColorDeconvolution',
                'mode': 'python',
                'script': codeToRun,
                'inputs': [{
                    'id': 'inputImageFile',
                    'type': 'string',
                    'format': 'string',
                    'target': 'filepath'
                }],
                'outputs': [{
                    'id': 'outputImageFile',
                    'type': 'string',
                    'format': 'string',
                    'target': 'filepath'
                }]
            },
            'inputs': {
                'inputImageFile': {
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
                'outputImageFile': {
                    "mode": "girder",
                    "parent_id": str(folder['_id']),
                    "name": 'out_' + item['name'],
                    "host": 'localhost',
                    "format": "string",
                    "type": "string",
                    'port': 8080,
                    'token': token,
                    'resource_type': 'item',
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
    info['apiRoot'].ColorDeconvolution = ColorDeconvolution()
