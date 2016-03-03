import os
import json
from lxml import etree

from girder.api.rest import Resource, loadmodel, boundHandler
from girder.api import access
from girder.api.describe import Description, describeRoute
from girder.constants import AccessType
from girder.plugins.worker import utils as wutils


def genHandlerToRunCLI(restResource, xmlFile, scriptFile):
    """Generates a handler to run CLI using girder_worker

    Parameters
    ----------
    restResource : girder.api.rest.Resource
        The object of a class derived from girder.api.rest.Resource to which
        this handler will be attached
    xmlFile : str
        Full path to xml file of the CLI
    scriptFile : str
        Full path to .py file containing the code of the clu

    Returns
    -------
    function
        Returns a function that runs the CLI using girder_worker
    """

    xmlPath, xmlNameWExt = os.path.split(xmlFile)
    xmlName = os.path.splitext(xmlNameWExt)[0]

    # parse xml of cli
    clixml = etree.parse(xmlFile)

    # read the script file containing the code into a string
    with open(scriptFile) as f:
        codeToRun = f.read()

    # do stuff needed to create REST endpoint for cLI
    handlerDesc = Description(clixml.findtext('title'))
    taskSpec = {'name': xmlName,
                'mode': 'python',
                'inputs': [],
                'outputs': []}

    slicerToGirderTypeMap = {
        'boolean': 'boolean',
        'integer': 'integer',
        'float': 'number',
        'double': 'number',
        'string': 'string',
        'integer-vector': 'integer_list',
        'float-vector': 'number_list',
        'double-vector': 'number_list',
        'string-vector': 'string_list',
        'integer-enumeration': 'integer',
        'float-enumeration': 'number',
        'double-enumeration': 'number',
        'string-enumeration': 'string',
        'file': 'string',
        'directory': 'string',
        'image': 'string'}

    inputGirderSuffix = '_girderId'
    outputGirderSuffix = '_folder_girderId'
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
            return [str(e.strip()) for e in strVal.split(',')]
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
        curName = elt.findtext('name')
        curType = elt.tag
        curDesc = elt.findtext('description')

        curTaskSpec = dict()
        curTaskSpec['id'] = curName
        curTaskSpec['type'] = slicerToGirderTypeMap[curType]
        curTaskSpec['format'] = slicerToGirderTypeMap[curType]

        if curType in ['image', 'file', 'directory']:
            curTaskSpec['target'] = 'filepath'  # check
            handlerDesc.param(curName + inputGirderSuffix,
                              'Girder ID of input %s - %s: %s'
                              % (curType, curName, curDesc),
                              dataType='string')
        else:
            handlerDesc.param(curName, curDesc, dataType='string')

        taskSpec['inputs'].append(curTaskSpec)

    # generate task spec for optional parameters
    for elt in paramXMLElements:
        curName = elt.findtext('name')
        curType = elt.tag
        curDesc = elt.findtext('description')

        curTaskSpec = dict()
        curTaskSpec['id'] = curName
        curTaskSpec['type'] = slicerToGirderTypeMap[curType]
        curTaskSpec['format'] = slicerToGirderTypeMap[curType]

        defaultValSpec = dict()
        defaultValSpec['format'] = curTaskSpec['format']
        strDefaultVal = elt.findtext('default')
        if strDefaultVal is not None:
            defaultVal = getDefaultValFromString(strDefaultVal,
                                                 curType)
        elif curType == 'boolean':
            defaultVal = False
        else:
            raise Exception(
                'optional parameters of type %s must '
                'provide a default value in the xml' % curType)
        defaultValSpec['data'] = defaultVal
        curTaskSpec['default'] = defaultValSpec

        handlerDesc.param(curName,
                          curDesc,
                          dataType='string',
                          required=False,
                          default=json.dumps(defaultVal))

        taskSpec['inputs'].append(curTaskSpec)

    # generate task spec for outputs
    for elt in outputXMLElements:
        curName = elt.findtext('name')
        curType = elt.tag
        curDesc = elt.findtext('description')

        # task spec for the current output
        curTaskSpec = dict()
        curTaskSpec['id'] = curName
        curTaskSpec['type'] = slicerToGirderTypeMap[curType]
        curTaskSpec['format'] = slicerToGirderTypeMap[curType]

        if curType in ['image', 'file', 'directory']:
            curTaskSpec['target'] = 'filepath'  # check

        taskSpec['outputs'].append(curTaskSpec)

        # param for parent folder
        handlerDesc.param(curName + outputGirderSuffix,
                          'Girder ID of parent folder '
                          'for output %s - %s: %s'
                          % (curType, curName, curDesc),
                          dataType='string')

        # param for name by which to store the current output
        handlerDesc.param(curName + outGirderNameSuffix,
                          'Name of output %s - %s: %s'
                          % (curType, curName, curDesc),
                          dataType='string')

    # define CLI handler function
    @boundHandler(restResource)
    @access.user
    @describeRoute(handlerDesc)
    def cliHandler(self, **args):

        user = self.getCurrentUser()
        token = self.getCurrentToken()['_id']

        # create job
        jobModel = self.model('job', 'jobs')
        jobTitle = '.'.join((restResource.resourceName, xmlName))
        job = jobModel.createJob(title=jobTitle,
                                 type=jobTitle,
                                 handler='worker_handler',
                                 user=user)
        kwargs = {
            'validate': False,
            'auto_convert': True,
            'cleanup': True}
        taskSpec['script'] = codeToRun

        # create job info
        jobToken = jobModel.createJobToken(job)
        kwargs['jobInfo'] = wutils.jobInfoSpec(job, jobToken)

        # create input bindings
        kwargs['inputs'] = dict()
        for elt in inputXMLElements:
            curName = elt.findtext('name')
            curType = elt.tag

            curBindingSpec = dict()
            if curType in ['image', 'file', 'directory']:
                # inputs of type image, file, or directory
                # should be located on girder
                if curType in ['image', 'file']:
                    curGirderType = 'item'
                else:
                    curGirderType = 'folder'
                curBindingSpec = wutils.girderInputSpec(
                    args[curName],
                    resourceType=curGirderType,
                    dataType='string', dataFormat='string',
                    token=token)
            else:
                # inputs that are not of type image, file, or directory
                # should be passed inline as string from json.dumps()
                curBindingSpec['mode'] = 'inline'
                curBindingSpec['type'] = slicerToGirderTypeMap[curType]
                curBindingSpec['format'] = 'json'
                curBindingSpec['data'] = args['params'][curName]

            kwargs['inputs'][curName] = curBindingSpec

        # create optional parameter bindings
        for elt in paramXMLElements:
            curName = elt.findtext('name')
            curType = elt.tag

            curBindingSpec = dict()
            curBindingSpec['mode'] = 'inline'
            curBindingSpec['type'] = slicerToGirderTypeMap[curType]
            curBindingSpec['format'] = 'json'
            curBindingSpec['data'] = args['params'][curName]

            kwargs['inputs'][curName] = curBindingSpec

        # create output boundings
        kwargs['outputs'] = dict()
        for elt in outputXMLElements:
            curName = elt.findtext('name')

            curBindingSpec = wutils.girderOutputSpec(
                args[curName],
                token,
                name=args['params'][curName + outGirderNameSuffix],
                dataType='string', dataFormat='string')
            kwargs['outputs'][curName] = curBindingSpec

            codeToSetOutputPath = "%s = os.path.join(_tempdir, '%s')" % (
                curName, args['params'][curName + outGirderNameSuffix])
            taskSpec['script'] = '\n'.join((codeToSetOutputPath,
                                           taskSpec['script']))

        # create task spec
        taskSpec['script'] = '\n'.join(("import os", taskSpec['script']))
        kwargs['task'] = taskSpec

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
            curModel = 'item'
        elif curType == 'directory':
            curModel = 'folder'
        else:
            continue
        curMap = {curName + inputGirderSuffix: curName}

        handlerFunc = loadmodel(map=curMap,
                                model=curModel,
                                level=AccessType.READ)(handlerFunc)

    # loadmodel stuff for outputs to girder
    for elt in outputXMLElements:
        curName = elt.findtext('name')

        curModel = 'folder'
        curMap = {curName + outputGirderSuffix: curName}

        handlerFunc = loadmodel(map=curMap,
                                model=curModel,
                                level=AccessType.WRITE)(handlerFunc)

    return handlerFunc


def genHandlerToGetCLIXmlSpec(restResource, xmlFile):
    """Generates a handler that returns the XML spec of the CLI

    Parameters
    ----------
    restResource : girder.api.rest.Resource
        The object of a class derived from girder.api.rest.Resource to which
        this handler will be attached
    xmlFile : str
        Full path to xml file of the CLI

    Returns
    -------
    function
        Returns a function that returns the xml spec of the CLI
    """

    xmlPath, xmlNameWExt = os.path.split(xmlFile)
    xmlName = os.path.splitext(xmlNameWExt)[0]

    # parse xml of cli
    clixml = etree.parse(xmlFile)

    # define the handler that returns the CLI's xml spec
    @boundHandler(restResource)
    @access.user
    @describeRoute(
        Description('Get XML spec of %s CLI' % xmlName)
    )
    def getXMLSpecHandler(self, *args, **kwargs):
        return etree.tostring(clixml)

    return getXMLSpecHandler


def genRESTEndPointsForSlicerCLIsInSubDirs(info, restResourceName, cliRootDir):
    """Generates REST end points for slicer CLIs placed in subdirectories of a
    given root directory and attaches them to a REST resource with the given
    name.

    For each CLI, it creates:
    * a GET Route (<apiURL>/`restResourceName`/<cliName>) that returns the
      xml spec of the CLI
    * a POST Route (<apiURL>/`restResourceName`/<cliName>/run) that runs the CLI

    It also creates a GET route (<apiURL>/`restResourceName`) that returns a
    list of relative routes to all CLIs attached to the generated REST resource

    Parameters
    ----------
    info
    restResourceName : str
        Name of the REST resource to which the end-points should be attached
    cliRootDir : str
        Full path of root directory containing the CLIs

    Returns
    -------

    """

    # create REST resource
    restResource = type(restResourceName,
                        (Resource, ),
                        {'resourceName': restResourceName})()

    # Add REST route for slicer CLIs located in subdirectories
    subdirList = [child
                  for child in os.listdir(cliRootDir)
                  if os.path.isdir(os.path.join(cliRootDir, child))]

    cliList = []

    for subdir in subdirList:

        # check if subdir contains a .xml file with the same name
        xmlFile = os.path.join(cliRootDir, subdir, subdir + '.xml')

        if not os.path.isfile(xmlFile):
            continue

        print xmlFile

        xmlPath, xmlNameWExt = os.path.split(xmlFile)
        xmlName = os.path.splitext(xmlNameWExt)[0]

        # TODO: check if the xml adheres to slicer execution model xml schema

        # check if subdir contains a .py file with the same name
        scriptFile = os.path.join(xmlPath, xmlName + '.py')

        if not os.path.isfile(scriptFile):
            continue

        # create a POST REST route that runs the CLI by invoking the handler
        try:
            cliRunHandler = genHandlerToRunCLI(restResource,
                                               xmlFile, scriptFile)
        except Exception as e:
            print "Failed to create REST endpoints for %s: %s" % (xmlName, e)
            continue

        cliRunHandlerName = 'run_' + xmlName
        setattr(restResource, cliRunHandlerName, cliRunHandler)
        restResource.route('POST',
                           (xmlName, 'run'),
                           getattr(restResource, cliRunHandlerName))

        # create GET REST route that returns the xml of the CLI
        try:
            cliGetXMLSpecHandler = genHandlerToGetCLIXmlSpec(restResource,
                                                             xmlFile)
        except Exception as e:
            print "Failed to create REST endpoints for %s: %s" % (xmlName, e)
            continue

        cliGetXMLSpecHandlerName = 'get_xml_' + xmlName
        setattr(restResource,
                cliGetXMLSpecHandlerName,
                cliGetXMLSpecHandler)
        restResource.route('GET',
                           (xmlName, 'xmlspec',),
                           getattr(restResource, cliGetXMLSpecHandlerName))

        cliList.append(xmlName)

    # create GET route that returns a list of relative routes to all CLIs
    @boundHandler(restResource)
    @access.user
    @describeRoute(
        Description('Get list of relative routes to all CLIs')
    )
    def getCLIListHandler(self, *args, **kwargs):
        return cliList

    getCLIListHandlerName = 'get_cli_list'
    setattr(restResource, getCLIListHandlerName, getCLIListHandler)
    restResource.route('GET', (), getattr(restResource, getCLIListHandlerName))

    # expose the generated REST resource via apiRoot
    setattr(info['apiRoot'], restResourceName, restResource)


def load(info):
    cliRootDir = os.path.dirname(__file__)
    genRESTEndPointsForSlicerCLIsInSubDirs(info, 'HistomicsTK', cliRootDir)
