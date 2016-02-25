import os
import sys
import json
from lxml import etree

from girder.api.rest import Resource, loadmodel, getApiUrl
from girder.api import access
from girder.api.describe import Description, describeRoute
from girder.constants import AccessType
from girder.plugins import worker


class HistomicsTK(Resource):
    def __init__(self):

        super(HistomicsTK, self).__init__()

        self.resourceName = 'HistomicsTK'

        self.route('POST',
                   ('ColorDeconvolution', 'run',),
                   self.runColorDeconvolution)

    @access.user
    @describeRoute(
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
    @loadmodel(map={'itemId': 'item'},
               model='item',
               level=AccessType.READ)
    @loadmodel(map={'folderId': 'folder'},
               model='folder',
               level=AccessType.WRITE)
    def runColorDeconvolution(self, item, folder, params):

        with open(os.path.join(os.path.dirname(__file__),
                               'ColorDeconvolution'
                               'ColorDeconvolution.py')) as f:
            codeToRun = f.read()

        jobModel = self.model('job', 'jobs')
        user = self.getCurrentUser()

        job = jobModel.createJob(title='ColorDeconvolution',
                                 type='ColorDeconvolution',
                                 handler='worker_handler',
                                 user=user)
        jobToken = jobModel.createJobToken(job)
        token = self.getCurrentToken()['_id']

        outFPrefix, outFSuffix = os.path.splitext(item['name'])
        outFPrefix = outFPrefix + '_'

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
                    'name': outFPrefix + 'stain_1' + outFSuffix,
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
                    'name': outFPrefix + 'stain_2' + outFSuffix,
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
                    'name': outFPrefix + 'stain_3' + outFSuffix,
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


def genRESTRouteForSlicerCLI(info, restResourceName):

    # create REST resource
    restResource = type(restResourceName,
                        (Resource, ),
                        {'resourceName': restResourceName})()

    # Add REST route for slicer CLIs located in subdirectories
    subdirList = filter(os.path.isdir, os.listdir('.'))

    for sdir in subdirList:
      
        # check if sdir contains a .xml file with the same name
        xmlFile = os.path.join('.', sdir, sdir + '.xml')

        if not os.path.isfile(xmlFile):
            continue

        xmlPath, xmlNameWExt = os.path.split(xmlFile)
        xmlName = os.path.splitext(xmlNameWExt)[0]

        # TODO: check if the xml adheres to slicer execution model
        
        # check if sdir contains a .py file with the same name
        scriptFile = os.path.join(xmlPath, xmlName + '.py')

        if not os.path.isfile(scriptFile):
            continue

        # parse xml of cli
        clixml = etree.parse(xmlFile)

        print xmlFile

        # read the script file containing the code into a string
        with open(scriptFile) as f:
            codeToRun = f.read()

        # create a handler to run the CLI using girder_worker
        def genCLIHandler():

            # do stuff needed to create REST endpoint for cLI
            handlerDesc = Description(clixml.find('description'))
            taskSpec = {'name': xmlName,
                        'mode': 'python',
                        'script': codeToRun,
                        'inputs': [],
                        'outputs': []}

            slicerToGirderTypeMap = {
                'boolean': 'boolean',
                'integer': 'integer',
                'float': 'number',
                'string': 'string',
                'integer-vector': 'integer-list',
                'float-vector': 'number-list',
                'double-vector': 'number-list',
                'string-vector': 'string-list',
                'integer-enumeration': 'integer',
                'float-enumeration': 'number',
                'double-enumeration': 'number',
                'string-enumeration': 'string',
                'file': 'string',
                'directory': 'string',
                'image': 'string'}

            inGirderSuffix = '_girderId'
            outGirderSuffix = '_girderFolderId'
            outGirderNameSuffix = '_name'

            def getDefaultValFromString(strVal, typeVal):
                if typeVal in ['integer', 'integer-enumeration']:
                    return int(strVal)
                elif typeVal in ['float', 'float-enumeration',
                                 'double', 'double-enumeration']:
                    return float(strVal)
                elif typeVal == 'integer-vector':
                    return [int(e.strip()) for e in strVal.split(',')]
                elif typeVal in ['float-vector', 'double-vector']:
                    return [float(e.strip()) for e in strVal.split(',')]
                elif typeVal == 'string-vector':
                    return [int(e.strip()) for e in strVal.split(',')]
                else:
                    return typeVal
            
            # identify xml elements of input, output, and optional params
            ioXMLElements = []
            paramXMLElements = []
            inputXMLElements = []
            outputXMLElements = []

            for pgelt in clixml.getiterator('parameters'):
                for pelt in pgelt:

                    if pelt.tag in ['description', 'label']:
                        continue

                    if pelt.tag not in slicerToGirderTypeMap.keys():
                        raise Exception(
                            'Parameter type %s is currently not supported' %
                            pelt.tag)

                    channel = pelt.findtext('channel')
                    if channel is not None:
                        if channel.lower() not in ['input', 'output']:
                            raise Exception('channel must be input or output.')
                        ioXMLElements.append(pelt)
                    else:
                        if pelt.tag in ['image', 'file', 'directory']:
                            raise Exception('optional parameters of type'
                                            'image, file, or directory are '
                                            'currently not supported')
                        paramXMLElements.append(pelt)
            
            ioXMLElements = sorted(ioXMLElements,
                                   key=lambda elt: elt.findtext('index'))

            for elt in ioXMLElements:
                if elt.findtext('channel').lower() == 'input':
                    inputXMLElements.append(elt)
                else:
                    if elt.tag not in ['image', 'file', 'directory']:
                        raise Exception(
                            'outputs of type other than image, file, or '
                            'directory are not currently supported.')
                    outputXMLElements.append(elt)

            # generate task spec for inputs
            for elt in inputXMLElements:
                curTaskSpec = {}
                curTaskSpec['id'] = elt.findtext('name')
                curTaskSpec['type'] = slicerToGirderTypeMap[elt.tag]
                curTaskSpec['format'] = slicerToGirderTypeMap[elt.tag]

                if elt.tag in ['image', 'file', 'directory']:
                    curTaskSpec['target'] = 'filepath'  # check
                    handlerDesc.param(elt.findtext('name') + inGirderSuffix,
                                      'Girder ID of input %s: '
                                      % elt.findtext('name') +
                                      elt.findtext('description'))
                else:
                    handlerDesc.param(elt.findtext('name'),
                                      elt.findtext('description'),
                                      dataType='string')

                taskSpec['inputs'].append(curTaskSpec)

            # generate task spec for outputs
            for elt in outputXMLElements:

                # task spec for the current output
                curTaskSpec = {}
                curTaskSpec['id'] = elt.findtext('name')
                curTaskSpec['type'] = slicerToGirderTypeMap[elt.tag]
                curTaskSpec['format'] = slicerToGirderTypeMap[elt.tag]

                if elt.tag in ['image', 'file', 'directory']:
                    curTaskSpec['target'] = 'filepath'

                taskSpec['outputs'].append(curTaskSpec)

                # param for parent folder
                handlerDesc.param(elt.findtext('name') + outGirderSuffix,
                                  'Girder ID of parent folder for output %s: '
                                  % elt.findtext('name') +
                                  elt.findtext('description'))

                # param for name by which to store the current output
                handlerDesc.param(elt.findtext('name') + outGirderNameSuffix,
                                  'Name of output %s: '
                                  % elt.findtext('name') +
                                  elt.findtext('description'))

            # generate task spec for optional parameters
            for elt in paramXMLElements:
                curTaskSpec = {}
                curTaskSpec['id'] = elt.findtext('name')
                curTaskSpec['type'] = slicerToGirderTypeMap[elt.tag]
                curTaskSpec['format'] = slicerToGirderTypeMap[elt.tag]
                
                defaultValSpec = {}
                defaultValSpec['format'] = curTaskSpec['format']
                strDefaultVal = elt.findtext('default')
                if strDefaultVal is not None:
                    defaultVal = getDefaultValFromString(strDefaultVal,
                                                         elt.tag)
                elif elt.tag == 'boolean':
                    defaultVal = False
                else:
                    raise Exception(
                        'All parameters of type other than boolean must '
                        'provide a default value in the xml')
                defaultValSpec['data'] = defaultVal
                curTaskSpec['default'] = defaultValSpec

                handlerDesc.param(elt.findtext('name'),
                                  elt.findtext('description'),
                                  dataType='string',
                                  required=False,
                                  default=json.dumps(defaultVal))

                taskSpec['inputs'].append(curTaskSpec)

            def cliHandler(self, **params):

                user = self.getCurrentUser()
                token = self.getCurrentToken()['_id']

                # create job
                jobModel = self.model('job', 'jobs')
                jobTitle = '.'.join((restResourceName, xmlName))
                job = jobModel.createJob(title=jobTitle,
                                         type=jobTitle,
                                         handler='worker_handler',
                                         user=user)
                kwargs = {}

                # create job info
                jobToken = jobModel.createJobToken(job)
                kwargs['jobInfo'] = worker.utils.jobInfoSpec(job, jobToken)
                
                # create task spec
                kwargs['task'] = taskSpec
                
                # create input bindings
                kwargs['inputs'] = {}
                for elt in inputXMLElements:
                    curName = elt.findtext('name')
                    curType = elt.tag

                    curBindingSpec = {}
                    if curType in ['image', 'file', 'directory']:
                        # inputs of type image, file, or directory
                        # should be located on girder
                        if curType in ['image', 'file']:
                            curGirderType = 'file'
                        else:
                            curGirderType = 'folder'
                        curBindingSpec = worker.utils.girderInputSpec(
                            params[curName], curGirderType, token=token)
                    else:
                        # inputs that are not of type image, file, or directory
                        # should be passed inline as string from json.dumps()
                        curBindingSpec['mode'] = 'inline'
                        curBindingSpec['type'] = slicerToGirderTypeMap[curType]
                        curBindingSpec['format'] = 'json'
                        curBindingSpec['data'] = params[curName]

                    kwargs['inputs'][curName] = curBindingSpec
                        
                # create output boundings
                kwargs['outputs'] = {}
                for elt in outputXMLElements:
                    curName = elt.findtext('name')
                    curType = elt.tag

                    if curType in ['image', 'file']:
                        curGirderType = 'file'
                    else:
                        curGirderType = 'folder'

                    curBindingSpec = worker.utils.girderOutputSpec(
                        params[curName], token,
                        name=params[curName + outGirderNameSuffix])

                    kwargs['outputs'][curName] = curBindingSpec

                # schedule job
                job['kwargs'] = kwargs
                job = jobModel.save(job)
                jobModel.scheduleJob(job)

                # return result
                return jobModel.filter(job, user)

            handlerFunc = cliHandler

            # loadmodel stuff for inputs in girder
            for elt in inputXMLElements:
                curName = elt.findtext('name')
                curType = elt.tag

                if curType in ['image', 'file']:
                    curModel = 'file'
                elif curType == 'directory':
                    curModel = 'folder'
                else:
                    continue
                curMap = {curName + inGirderSuffix: curName}

                handlerFunc = loadmodel(map=curMap,
                                        model=curModel,
                                        level=AccessType.READ)(handlerFunc)

            # loadmodel stuff for outputs to girder
            for elt in outputXMLElements:
                curName = elt.findtext('name')
                curType = elt.tag

                curModel = 'folder'
                curMap = {curName + outGirderSuffix: curName}

                handlerFunc = loadmodel(map=curMap,
                                        model=curModel,
                                        level=AccessType.WRITE)(handlerFunc)

            # add route description to the handler
            handlerFunc = describeRoute(handlerDesc)(handlerFunc)

            # add user access
            handlerFunc = access.user(handlerFunc)

            print taskSpec

            return handlerFunc

        # create a POST REST route that runs the CLI by invoking the handler
        try:
            cliHandlerFunc = genCLIHandler()
        except:
            e = sys.exec_info()[0]
            print "Failed to create REST endpoints for %s" % xmlPath
            print e
            continue

        cliHandlerName = 'run_' + xmlName
        setattr(restResource, cliHandlerName, cliHandlerFunc)
        restResource.route('POST',
                           (xmlName, 'run'),
                           getattr(restResource, cliHandlerName))

        # create GET REST route that returns the xml of the CLI
        def getXMLSpec(self, **params):
            return etree.tostring(clixml)

        cliGetXMLSpecHandlerName = 'get_xml_' + xmlName
        setattr(restResource,
                cliGetXMLSpecHandlerName,
                access.user(getXMLSpec))
        restResource.route('GET',
                           (xmlName, 'getXMLSpec'),
                           getattr(restResource, cliGetXMLSpecHandlerName))

    # expose the generated REST resource via apiRoot
    setattr(info['apiRoot'], restResourceName, restResource)


def load(info):
    info['apiRoot'].HistomicsTK = HistomicsTK()
    # genRESTRouteForSlicerCLI(info, 'HistomicsTK')

