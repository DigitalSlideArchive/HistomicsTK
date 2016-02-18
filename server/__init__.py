import os
from girder.api.rest import Resource, loadmodel, getApiUrl
from girder.api import access
from girder.api.describe import Description
from girder.constants import AccessType

from lxml import etree
from io import StringIO

 
class ColorDeconvolution(Resource):
    def __init__(self):
        super(ColorDeconvolution, self).__init__()
        self.resourceName = 'ColorDeconvolution'
        self.route('POST', ('run',), self.runAnalysis)

    @access.user
    @loadmodel(map={'itemId': 'item'},
               model='item',
               level=AccessType.READ)
    @loadmodel(map={'folderId': 'folder'},
               model='folder',
               level=AccessType.WRITE)
    def runAnalysis(self, item, folder,
                    params):
        with open(os.path.join(os.path.dirname(__file__), 'script.py')) as f:
            codeToRun = f.read()

        jobModel = self.model('job', 'jobs')
        user = self.getCurrentUser()

        job = jobModel.createJob(title='ColorDeconvolution',
                                 type='ColorDeconvolution',
                                 handler='worker_handler',
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
                }, {
                    'id': 'stainColor_1',
                    'type': 'number_list',
                    'format': 'number_list'
                }, {
                    'id': 'stainColor_2',
                    'type': 'number_list',
                    'format': 'number_list'
                }, {
                    'id': 'stainColor_3',
                    'type': 'number_list',
                    'format': 'number_list',
                    'default': {
                        'format': 'number_list',
                        'data': [0, 0, 0]
                        }
                }],
                'outputs': [{
                    'id': 'outputStainImageFile_1',
                    'type': 'string',
                    'format': 'string',
                    'target': 'filepath'
                }, {
                    'id': 'outputStainImageFile_2',
                    'type': 'string',
                    'format': 'string',
                    'target': 'filepath'
                }, {
                    'id': 'outputStainImageFile_3',
                    'type': 'string',
                    'format': 'string',
                    'target': 'filepath'
                }]
            },
            'inputs': {
                'inputImageFile': {
                    'mode': 'girder',
                    'id': str(item['_id']),
                    'name': item['name'],
                    'host': 'localhost',
                    'format': 'string',
                    'type': 'string',
                    'port': 8080,
                    'token': token,
                    'resource_type': 'item'
                },
                'stainColor_1': {
                    'mode': 'inline',
                    'type': 'number_list',
                    'format': 'json',
                    'data': params['stainColor_1']
                },
                'stainColor_2': {
                    'mode': 'inline',
                    'type': 'number_list',
                    'format': 'json',
                    'data': params['stainColor_2']
                },
                'stainColor_3': {
                    'mode': 'inline',
                    'type': 'number_list',
                    'format': 'json',
                    'data': params['stainColor_3']
                }
            },
            'outputs': {
                'outputStainImageFile_1': {
                    'mode': 'girder',
                    'parent_id': str(folder['_id']),
                    'name': 'stain_1_' + item['name'],
                    'host': 'localhost',
                    'format': 'string',
                    'type': 'string',
                    'port': 8080,
                    'token': token,
                    'resource_type': 'item',
                    'parent_type': 'folder'
                },
                'outputStainImageFile_2': {
                    'mode': 'girder',
                    'parent_id': str(folder['_id']),
                    'name': 'stain_2_' + item['name'],
                    'host': 'localhost',
                    'format': 'string',
                    'type': 'string',
                    'port': 8080,
                    'token': token,
                    'resource_type': 'item',
                    'parent_type': 'folder'
                },
                'outputStainImageFile_3': {
                    'mode': 'girder',
                    'parent_id': str(folder['_id']),
                    'name': 'stain_3_' + item['name'],
                    'host': 'localhost',
                    'format': 'string',
                    'type': 'string',
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
            'auto_convert': True,
            'cleanup': True
        }
        job['kwargs'] = kwargs
        job = jobModel.save(job)
        jobModel.scheduleJob(job)

        return jobModel.filter(job, user)

    runAnalysis.description = (
        Description('Run color deconvolution on an image.')
        .param('itemId', 'ID of the item containing the image.')
        .param('folderId', 'ID of the output folder.')
        .param('stainColor_1',
               'A 3-element list containing the RGB color values of stain-1',
               dataType='string')
        .param('stainColor_2',
               'A 3-element list specifying the RGB color values of stain-2',
               dataType='string')
        .param('stainColor_3',
               'A 3-element list specifying the RGB color values of stain-3',
               dataType='string', required=False, default="[0, 0, 0]"))


def genRESTResourceFromSlicerXML(xml_spec_file, script_file=None):
    return ColorDeconvolution()


def load(info):
    info['apiRoot'].ColorDeconvolution = ColorDeconvolution()

    '''
    subdir_list = filter(os.path.isdir, os.listdir('.'))

    for sdir in subdir_list:
      
        xml_spec_file = os.path.join('.', sdir, sdir + '.xml')

        # check if sdir contains a .xml file with the same name
        if not os.path.isfile(xml_spec_file):
            continue
        
        # create rest route
        setattr(info['apiRoot'],
                sdir,
                genRESTResourceFromSlicerXML(xml_spec_file))
    '''
