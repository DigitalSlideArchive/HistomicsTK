import os
import json
from ctk_cli import CLIModule

from girder.api.rest import Resource, loadmodel, boundHandler
from girder.api import access
from girder.api.describe import Description, describeRoute
from girder.constants import AccessType
from girder.plugins.worker import utils as wutils

_SLICER_TO_GIRDER_WORKER_TYPE_MAP = {
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

_girderInputFileSuffix = '_girderId'
_girderOutputFolderSuffix = '_girderFolderId'
_girderOutputNameSuffix = '_name'


def getCLIParameters(clim):

    # get parameters
    index_params, opt_params, simple_out_params = clim.classifyParameters()

    # perform sanity checks
    if len(simple_out_params) > 0:
        raise Exception(
            'outputs of type other than image, file, or '
            'directory are not currently supported.')

    for param in index_params + opt_params:
        if param.typ not in _SLICER_TO_GIRDER_WORKER_TYPE_MAP.keys():
            raise Exception(
                'Parameter type %s is currently not supported' % param.type
            )

    for param in opt_params:
        if param.typ in ['image', 'file', 'directory']:
            raise Exception('optional parameters of type'
                            'image, file, or directory are '
                            'currently not supported')

    # sort indexed parameters in increasing order of index
    index_params.sort(key=lambda p: p.index)

    # sort opt parameters in increasing order of name for easy lookup
    def get_flag(p):
        if p.flag is not None:
            return p.flag.strip('-')
        elif p.longflag is not None:
            return p.longflag.strip('-')
        else:
            return None

    opt_params.sort(key=lambda p: get_flag(p))

    return index_params, opt_params


def createInputBindingSpec(param, hargs, token):

    curBindingSpec = dict()
    if param.typ in ['image', 'file', 'directory']:
        # inputs of type image, file, or directory are assumed to be on girder
        if param.typ in ['image', 'file']:
            curGirderType = 'item'
        else:
            curGirderType = 'folder'

        curBindingSpec = wutils.girderInputSpec(
            hargs[param.name],
            resourceType=curGirderType,
            dataType='string', dataFormat='string',
            token=token)
    else:
        # inputs that are not of type image, file, or directory
        # should be passed inline as string from json.dumps()
        curBindingSpec['mode'] = 'inline'
        curBindingSpec['type'] = _SLICER_TO_GIRDER_WORKER_TYPE_MAP[param.typ]
        curBindingSpec['format'] = 'json'
        curBindingSpec['data'] = hargs['params'][param.name]

    return curBindingSpec


def createOutputBindingSpec(param, hargs, token):

    curBindingSpec = wutils.girderOutputSpec(
        hargs[param.name],
        token,
        name=hargs['params'][param.name + _girderOutputNameSuffix],
        dataType='string', dataFormat='string'
    )

    return curBindingSpec


def createParamBindingSpec(param, hargs):

    curBindingSpec = dict()
    curBindingSpec['mode'] = 'inline'
    curBindingSpec['type'] = _SLICER_TO_GIRDER_WORKER_TYPE_MAP[param.typ]
    curBindingSpec['format'] = 'json'
    curBindingSpec['data'] = hargs['params'][param.name]

    return curBindingSpec


def createIndexedParamTaskSpec(param):
    """Creates task spec for indexed parameters

    Parameters
    ----------
    param : ctk_cli.CLIParameter
        parameter for which the task spec should be created

    """

    curTaskSpec = dict()
    curTaskSpec['id'] = param.name
    curTaskSpec['type'] = _SLICER_TO_GIRDER_WORKER_TYPE_MAP[param.typ]
    curTaskSpec['format'] = _SLICER_TO_GIRDER_WORKER_TYPE_MAP[param.typ]

    if param.typ in ['image', 'file', 'directory']:
        curTaskSpec['target'] = 'filepath'  # check

    return curTaskSpec


def addIndexedInputParams(index_input_params, taskSpec, handlerDesc):

    for param in index_input_params:

        # add to task spec
        curTaskSpec = createIndexedParamTaskSpec(param)
        taskSpec['inputs'].append(curTaskSpec)

        # add to route description
        if param.typ in ['image', 'file', 'directory']:
            handlerDesc.param(param.name + _girderInputFileSuffix,
                              'Girder ID of input %s - %s: %s'
                              % (param.typ, param.name, param.description),
                              dataType='string', required=True)
        else:
            handlerDesc.param(param.name, param.description,
                              dataType='string', required=True)


def addIndexedOutputParams(index_output_params, taskSpec, handlerDesc):

    for param in index_output_params:

        # add to task spec
        curTaskSpec = createIndexedParamTaskSpec(param)
        taskSpec['outputs'].append(curTaskSpec)

        # add param for parent folder to route description
        handlerDesc.param(param.name + _girderOutputFolderSuffix,
                          'Girder ID of parent folder '
                          'for output %s - %s: %s'
                          % (param.typ, param.typ, param.description),
                          dataType='string', required=True)

        # add param for name of current output to route description
        handlerDesc.param(param.name + _girderOutputNameSuffix,
                          'Name of output %s - %s: %s'
                          % (param.typ, param.name, param.description),
                          dataType='string', required=True)


def createOptionalParamTaskSpec(param):
    """Creates task spec for optional parameters

    Parameters
    ----------
    param : ctk_cli.CLIParameter
        parameter for which the task spec should be created

    """

    curTaskSpec = dict()
    curTaskSpec['id'] = param.name
    curTaskSpec['type'] = _SLICER_TO_GIRDER_WORKER_TYPE_MAP[param.typ]
    curTaskSpec['format'] = _SLICER_TO_GIRDER_WORKER_TYPE_MAP[param.typ]

    if param.typ in ['image', 'file', 'directory']:
        curTaskSpec['target'] = 'filepath'  # check

    defaultValSpec = dict()
    defaultValSpec['format'] = curTaskSpec['format']

    if param.default is not None:
        defaultValSpec['data'] = param.default
    elif param.typ == 'boolean':
        defaultValSpec['data'] = False
    elif param.typ in ['image', 'file', 'directory']:
        defaultValSpec['data'] = ''
    else:
        raise Exception(
            'optional parameters of type %s must '
            'provide a default value in the xml' % param.typ)
    curTaskSpec['default'] = defaultValSpec

    return curTaskSpec


def addOptionalParams(opt_params, taskSpec, handlerDesc):

    for param in opt_params:

        # add to task spec
        curTaskSpec = createOptionalParamTaskSpec(param)
        taskSpec['inputs'].append(curTaskSpec)

        # add to route description
        defaultVal = curTaskSpec['default']['data']
        handlerDesc.param(param.name, param.description,
                          dataType='string',
                          default=json.dumps(defaultVal),
                          required=False)


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

    print xmlName

    # parse cli xml spec
    clim = CLIModule(xmlFile)

    # read the script file containing the code into a string
    with open(scriptFile) as f:
        codeToRun = f.read()

    # create CLI description string
    str_description = ['Description: <br/><br/>' + clim.description]

    if clim.version is not None and len(clim.version) > 0:
        str_description.append('Version: ' + clim.version)

    if clim.license is not None and len(clim.license) > 0:
        str_description.append('License: ' + clim.license)

    if clim.contributor is not None and len(clim.contributor) > 0:
        str_description.append('Author(s): ' + clim.contributor)

    if clim.acknowledgements is not None and \
       len(clim.acknowledgements) > 0:
        str_description.append(
            'Acknowledgements: ' + clim.acknowledgements)

    str_description = '<br/><br/>'.join(str_description)

    # do stuff needed to create REST endpoint for cLI
    handlerDesc = Description(clim.title).notes(str_description)
    taskSpec = {'name': xmlName,
                'mode': 'python',
                'inputs': [],
                'outputs': []}

    # get CLI parameters
    index_params, opt_params = getCLIParameters(clim)

    index_input_params = filter(lambda p: p.channel == 'input', index_params)
    index_output_params = filter(lambda p: p.channel == 'output', index_params)

    # generate task spec for indexed input parameters
    addIndexedInputParams(index_input_params, taskSpec, handlerDesc)

    # generate task spec for indexed output parameters
    addIndexedOutputParams(index_output_params, taskSpec, handlerDesc)

    # generate task spec for optional parameters
    addOptionalParams(opt_params, taskSpec, handlerDesc)

    # define CLI handler function
    @boundHandler(restResource)
    @access.user
    @describeRoute(handlerDesc)
    def cliHandler(self, **hargs):

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

        print hargs

        # create job info
        jobToken = jobModel.createJobToken(job)
        kwargs['jobInfo'] = wutils.jobInfoSpec(job, jobToken)

        # create indexed input parameter bindings
        kwargs['inputs'] = dict()
        for param in index_input_params:
            curBindingSpec = createInputBindingSpec(param, hargs, token)
            kwargs['inputs'][param.name] = curBindingSpec

        # create optional parameter bindings
        for param in opt_params:
            curBindingSpec = createParamBindingSpec(param, hargs)
            kwargs['inputs'][param.name] = curBindingSpec

        # create indexed output boundings
        kwargs['outputs'] = dict()
        for param in index_output_params:
            curBindingSpec = createOutputBindingSpec(param, hargs, token)
            kwargs['outputs'][param.name] = curBindingSpec

        # point output file variables to actual paths on the machine
        for param in index_output_params:
            codeToSetOutputPath = "%s = os.path.join(_tempdir, '%s')" % (
                param.name,
                hargs['params'][param.name + _girderOutputNameSuffix])
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

    # loadmodel stuff for indexed inputs in girder
    for param in index_input_params:
        if param.typ in ['image', 'file']:
            curModel = 'item'
        elif param.typ == 'directory':
            curModel = 'folder'
        else:
            continue
        curMap = {param.name + _girderInputFileSuffix: param.name}

        handlerFunc = loadmodel(map=curMap,
                                model=curModel,
                                level=AccessType.READ)(handlerFunc)

    # loadmodel stuff for indexed outputs to girder
    for param in index_output_params:
        curModel = 'folder'
        curMap = {param.name + _girderOutputFolderSuffix: param.name}

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

    # read xml into a string
    with open(xmlFile) as f:
        str_xml = f.read()

    # define the handler that returns the CLI's xml spec
    @boundHandler(restResource)
    @access.user
    @describeRoute(
        Description('Get XML spec of %s CLI' % xmlName)
    )
    def getXMLSpecHandler(self, *args, **kwargs):
        return str_xml

    return getXMLSpecHandler


def genRESTEndPointsForSlicerCLIsInSubDirs(info, restResourceName, cliRootDir):
    """Generates REST end points for slicer CLIs placed in subdirectories of a
    given root directory and attaches them to a REST resource with the given
    name.

    For each CLI, it creates:
    * a GET Route (<apiURL>/`restResourceName`/<cliRelativePath>/xmlspec)
    that returns the xml spec of the CLI
    * a POST Route (<apiURL>/`restResourceName`/<cliRelativePath>/run)
    that runs the CLI

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
    cliList = []

    for parentdir, dirnames, filenames in os.walk(cliRootDir):
        for subdir in dirnames:
            curCLIRelPath = os.path.relpath(os.path.join(parentdir, subdir),
                                            cliRootDir)

            # check if subdir contains a .xml file with the same name
            xmlFile = os.path.join(parentdir, subdir, subdir + '.xml')

            if not os.path.isfile(xmlFile):
                continue

            xmlPath, xmlNameWExt = os.path.split(xmlFile)
            xmlName = os.path.splitext(xmlNameWExt)[0]

            # check if subdir contains a .py file with the same name
            scriptFile = os.path.join(xmlPath, xmlName + '.py')

            if not os.path.isfile(scriptFile):
                continue

            # print curCLIRelPath

            # TODO: check if xml adheres to slicer execution model xml schema

            # create a POST REST route that runs the CLI
            try:
                cliRunHandler = genHandlerToRunCLI(restResource,
                                                   xmlFile, scriptFile)
            except Exception as e:
                print "Failed to create REST endpoints for %s: %s" % (
                    curCLIRelPath, e)
                continue

            cliRunHandlerName = 'run_' + xmlName
            setattr(restResource, cliRunHandlerName, cliRunHandler)
            restResource.route('POST',
                               (curCLIRelPath, 'run'),
                               getattr(restResource, cliRunHandlerName))

            # create GET REST route that returns the xml of the CLI
            try:
                cliGetXMLSpecHandler = genHandlerToGetCLIXmlSpec(restResource,
                                                                 xmlFile)
            except Exception as e:
                print "Failed to create REST endpoints for %s: %s" % (
                    curCLIRelPath, e)
                continue

            cliGetXMLSpecHandlerName = 'get_xml_' + xmlName
            setattr(restResource,
                    cliGetXMLSpecHandlerName,
                    cliGetXMLSpecHandler)
            restResource.route('GET',
                               (curCLIRelPath, 'xmlspec',),
                               getattr(restResource, cliGetXMLSpecHandlerName))

            cliList.append(curCLIRelPath)

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
