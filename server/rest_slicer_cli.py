import os
import json
import subprocess
from ctk_cli import CLIModule

from girder.api.rest import Resource, loadmodel, boundHandler
from girder.api import access
from girder.api.describe import Description, describeRoute
from girder.constants import AccessType
from girder.plugins.worker import utils as wutils
from girder.utility.model_importer import ModelImporter

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

_SLICER_TYPE_TO_GIRDER_MODEL_MAP = {
    'image': 'item',
    'file': 'item',
    'directory': 'folder'
}

_girderInputFileSuffix = '_girderItemId'
_girderOutputFolderSuffix = '_girderFolderId'
_girderOutputNameSuffix = '_name'

_return_parameter_file_name = 'returnparameterfile'


def _getCLIParameters(clim):

    # get parameters
    index_params, opt_params, simple_out_params = clim.classifyParameters()

    # perform sanity checks
    for param in index_params + opt_params:
        if param.typ not in _SLICER_TO_GIRDER_WORKER_TYPE_MAP.keys():
            raise Exception(
                'Parameter type %s is currently not supported' % param.typ
            )

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

    return index_params, opt_params, simple_out_params


def _createIndexedParamTaskSpec(param):
    """Creates task spec for indexed parameters

    Parameters
    ----------
    param : ctk_cli.CLIParameter
        parameter for which the task spec should be created

    """

    curTaskSpec = dict()
    curTaskSpec['id'] = param.name
    curTaskSpec['name'] = param.label
    curTaskSpec['type'] = _SLICER_TO_GIRDER_WORKER_TYPE_MAP[param.typ]
    curTaskSpec['format'] = _SLICER_TO_GIRDER_WORKER_TYPE_MAP[param.typ]

    if param.isExternalType():
        curTaskSpec['target'] = 'filepath'  # check

    return curTaskSpec


def _addIndexedInputParams(index_input_params, taskSpec, handlerDesc):

    for param in index_input_params:

        # add to task spec
        curTaskSpec = _createIndexedParamTaskSpec(param)
        taskSpec['inputs'].append(curTaskSpec)

        # add to route description
        if param.isExternalType():
            handlerDesc.param(param.name + _girderInputFileSuffix,
                              'Girder ID of input %s - %s: %s'
                              % (param.typ, param.name, param.description),
                              dataType='string', required=True)
        else:
            handlerDesc.param(param.name, param.description,
                              dataType='string', required=True)


def _addIndexedOutputParams(index_output_params, taskSpec, handlerDesc):

    for param in index_output_params:

        # add to task spec
        curTaskSpec = _createIndexedParamTaskSpec(param)
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


def _createOptionalParamTaskSpec(param):
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

    if param.isExternalType():
        curTaskSpec['target'] = 'filepath'  # check

    if param.channel != 'output':

        defaultValSpec = dict()
        defaultValSpec['format'] = curTaskSpec['format']

        if param.default is not None:
            defaultValSpec['data'] = param.default
        elif param.typ == 'boolean':
            defaultValSpec['data'] = False
        elif param.isExternalType():
            defaultValSpec['data'] = ""
        else:
            raise Exception(
                'optional parameters of type %s must '
                'provide a default value in the xml' % param.typ)
        curTaskSpec['default'] = defaultValSpec

    return curTaskSpec


def _addOptionalInputParams(opt_input_params, taskSpec, handlerDesc):

    for param in opt_input_params:

        # add to task spec
        curTaskSpec = _createOptionalParamTaskSpec(param)
        taskSpec['inputs'].append(curTaskSpec)

        # add to route description
        defaultVal = curTaskSpec['default']['data']

        if param.isExternalType():
            handlerDesc.param(param.name + _girderInputFileSuffix,
                              'Girder ID of input %s - %s: %s'
                              % (param.typ, param.name, param.description),
                              dataType='string',
                              required=False)
        else:
            handlerDesc.param(param.name, param.description,
                              dataType='string',
                              default=json.dumps(defaultVal),
                              required=False)


def _addOptionalOutputParams(opt_output_params, taskSpec, handlerDesc):

    for param in opt_output_params:

        if not param.isExternalType():
            continue

        # add to task spec
        curTaskSpec = _createOptionalParamTaskSpec(param)
        taskSpec['outputs'].append(curTaskSpec)

        # add param for parent folder to route description
        handlerDesc.param(param.name + _girderOutputFolderSuffix,
                          'Girder ID of parent folder '
                          'for output %s - %s: %s'
                          % (param.typ, param.name, param.description),
                          dataType='string',
                          required=False)

        # add param for name of current output to route description
        handlerDesc.param(param.name + _girderOutputNameSuffix,
                          'Name of output %s - %s: %s'
                          % (param.typ, param.name, param.description),
                          dataType='string', required=False)


def _addReturnParameterFileParam(taskSpec, handlerDesc):

    curName = _return_parameter_file_name
    curType = 'file'
    curDesc = """
        Filename in which to write simple return parameters
        (integer, float, integer-vector, etc.) as opposed to bulk
        return parameters (image, file, directory, geometry,
        transform, measurement, table).
    """

    # add to task spec
    curTaskSpec = dict()
    curTaskSpec['id'] = curName
    curTaskSpec['type'] = _SLICER_TO_GIRDER_WORKER_TYPE_MAP[curType]
    curTaskSpec['format'] = _SLICER_TO_GIRDER_WORKER_TYPE_MAP[curType]
    curTaskSpec['target'] = 'filepath'  # check
    taskSpec['outputs'].append(curTaskSpec)

    # add param for parent folder to route description
    handlerDesc.param(curName + _girderOutputFolderSuffix,
                      'Girder ID of parent folder '
                      'for output %s - %s: %s'
                      % (curType, curName, curDesc),
                      dataType='string',
                      required=False)

    # add param for name of current output to route description
    handlerDesc.param(curName + _girderOutputNameSuffix,
                      'Name of output %s - %s: %s'
                      % (curType, curName, curDesc),
                      dataType='string', required=False)


def _createInputParamBindingSpec(param, hargs, token):

    curBindingSpec = dict()
    if _is_on_girder(param):
        curBindingSpec = wutils.girderInputSpec(
            hargs[param.name],
            resourceType=_SLICER_TYPE_TO_GIRDER_MODEL_MAP[param.typ],
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


def _createOutputParamBindingSpec(param, hargs, token):

    curBindingSpec = wutils.girderOutputSpec(
        hargs[param.name],
        token,
        name=hargs['params'][param.name + _girderOutputNameSuffix],
        dataType='string', dataFormat='string'
    )

    return curBindingSpec


def _addIndexedInputParamBindings(index_input_params, bspec, hargs, token):

    for param in index_input_params:
        bspec[param.name] = _createInputParamBindingSpec(param, hargs, token)


def _addIndexedOutputParamBindings(index_output_params, bspec, hargs, token):

    for param in index_output_params:
        bspec[param.name] = _createOutputParamBindingSpec(param, hargs, token)


def _addOptionalInputParamBindings(opt_input_params, bspec, hargs, user, token):

    for param in opt_input_params:

        if _is_on_girder(param):
            if param.name + _girderInputFileSuffix not in hargs['params']:
                continue

            curModelName = _SLICER_TYPE_TO_GIRDER_MODEL_MAP[param.typ]
            curModel = ModelImporter.model(curModelName)
            curId = hargs['params'][param.name + _girderInputFileSuffix]

            hargs[param.name] = curModel.load(id=curId,
                                              level=AccessType.READ,
                                              user=user)

        bspec[param.name] = _createInputParamBindingSpec(param, hargs, token)


def _addOptionalOutputParamBindings(opt_output_params,
                                    bspec, hargs, user, token):

    for param in opt_output_params:

        if not _is_on_girder(param):
            continue

        # check if it was requested in the REST request
        if (param.name + _girderOutputFolderSuffix not in hargs['params'] or
                param.name + _girderOutputNameSuffix not in hargs['params']):
            continue

        curModel = ModelImporter.model('folder')
        curId = hargs['params'][param.name + _girderOutputFolderSuffix]

        hargs[param.name] = curModel.load(id=curId,
                                          level=AccessType.WRITE,
                                          user=user)

        bspec[param.name] = _createOutputParamBindingSpec(param, hargs, token)


def _addReturnParameterFileBinding(bspec, hargs, user, token):

    curName = _return_parameter_file_name

    if (curName + _girderOutputFolderSuffix not in hargs['params'] or
            curName + _girderOutputNameSuffix not in hargs['params']):
        return

    curModel = ModelImporter.model('folder')
    curId = hargs['params'][curName + _girderOutputFolderSuffix]

    hargs[curName] = curModel.load(id=curId,
                                   level=AccessType.WRITE,
                                   user=user)

    curBindingSpec = wutils.girderOutputSpec(
        hargs[curName],
        token,
        name=hargs['params'][curName + _girderOutputNameSuffix],
        dataType='string', dataFormat='string'
    )

    bspec[curName] = curBindingSpec


def _addCodeToSetIndexOutputFileParams(index_output_params, taskSpec, hargs):
    for param in index_output_params:
        codeToSetOutputPath = "%s = os.path.join(_tempdir, '%s')" % (
            param.name,
            hargs['params'][param.name + _girderOutputNameSuffix])
        taskSpec['script'] = '\n'.join((codeToSetOutputPath,
                                       taskSpec['script']))


def _addCodeToSetOptOutputFileParams(opt_output_params,
                                     kwargs, taskSpec, hargs):

    for param in opt_output_params:

        if not _is_on_girder(param):
            continue

        if param.name in kwargs['outputs']:
            codeToSetOutputPath = "%s = os.path.join(_tempdir, '%s')" % (
                param.name,
                hargs['params'][param.name + _girderOutputNameSuffix]
            )
        else:
            codeToSetOutputPath = "%s = None" % param.name

        taskSpec['script'] = '\n'.join((codeToSetOutputPath,
                                       taskSpec['script']))


def _addCodeToSetReturnParameterFile(kwargs, taskSpec, hargs):

    curName = _return_parameter_file_name

    if curName in kwargs['outputs']:
        codeToSetOutputPath = "%s = os.path.join(_tempdir, '%s')" % (
            curName,
            hargs['params'][curName + _girderOutputNameSuffix]
        )
    else:
        codeToSetOutputPath = "%s = None" % curName

    taskSpec['script'] = '\n'.join((codeToSetOutputPath,
                                   taskSpec['script']))


def _is_on_girder(param):
    return param.typ in _SLICER_TYPE_TO_GIRDER_MODEL_MAP


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
    index_params, opt_params, simple_out_params = _getCLIParameters(clim)

    # add indexed input parameters
    index_input_params = filter(lambda p: p.channel == 'input', index_params)

    _addIndexedInputParams(index_input_params, taskSpec, handlerDesc)

    # add indexed output parameters
    index_output_params = filter(lambda p: p.channel == 'output', index_params)

    _addIndexedOutputParams(index_output_params, taskSpec, handlerDesc)

    # add optional input parameters
    opt_input_params = filter(lambda p: p.channel != 'output', opt_params)

    _addOptionalInputParams(opt_input_params, taskSpec, handlerDesc)

    # add optional output parameters
    opt_output_params = filter(lambda p: p.channel == 'output', opt_params)

    _addOptionalOutputParams(opt_output_params, taskSpec, handlerDesc)

    # add returnparameterfile if there are simple output params
    if len(simple_out_params) > 0:
        _addReturnParameterFileParam(taskSpec, handlerDesc)

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
            'cleanup': True,
            'inputs': dict(),
            'outputs': dict()
        }
        taskSpec['script'] = codeToRun

        # create job info
        jobToken = jobModel.createJobToken(job)
        kwargs['jobInfo'] = wutils.jobInfoSpec(job, jobToken)

        # create indexed input parameter bindings
        _addIndexedInputParamBindings(index_input_params,
                                      kwargs['inputs'], hargs, token)

        # create indexed output boundings
        _addIndexedOutputParamBindings(index_output_params,
                                       kwargs['outputs'], hargs, token)

        # create optional input parameter bindings
        _addOptionalInputParamBindings(opt_input_params,
                                       kwargs['inputs'], hargs, user, token)

        # create optional output parameter bindings
        _addOptionalOutputParamBindings(opt_output_params,
                                        kwargs['outputs'], hargs, user, token)

        # create returnparameterfile binding
        _addReturnParameterFileBinding(kwargs['outputs'], hargs, user, token)

        # point output file variables to actual paths
        _addCodeToSetIndexOutputFileParams(index_output_params, taskSpec, hargs)

        _addCodeToSetOptOutputFileParams(opt_output_params,
                                         kwargs, taskSpec, hargs)

        _addCodeToSetReturnParameterFile(kwargs, taskSpec, hargs)

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

    # loadmodel stuff for indexed input params on girder
    index_input_params_on_girder = filter(_is_on_girder, index_input_params)

    for param in index_input_params_on_girder:

        curModel = _SLICER_TYPE_TO_GIRDER_MODEL_MAP[param.typ]
        curMap = {param.name + _girderInputFileSuffix: param.name}

        handlerFunc = loadmodel(map=curMap,
                                model=curModel,
                                level=AccessType.READ)(handlerFunc)

    # loadmodel stuff for indexed output params on girder
    index_output_params_on_girder = filter(_is_on_girder, index_output_params)

    for param in index_output_params_on_girder:

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


def getParamCommandLineValue(param, value):
    if param.isVector():
        return ','.join(map(str, value))
    else:
        return str(value)


def _addOptionalInputParamsToContainerArgs(opt_input_params,
                                           containerArgs, hargs):

    for param in opt_input_params:

        if param.longflag:
            curFlag = '--' + param.longflag
        elif param.flag:
            curFlag = '-' + param.flag
        else:
            continue

        if _is_on_girder(param) and param.name in hargs:
            curValue = hargs[param.name]
        elif param.name in hargs['params']:
            curValue = getParamCommandLineValue(param,
                                                hargs['params'][param.name])
        else:
            continue

        containerArgs.append(curFlag, curValue)


def _addOptionalOutputParamsToContainerArgs(opt_output_params,
                                            containerArgs, kwargs, hargs):

    for param in opt_output_params:

        if param.longflag:
            curFlag = '--' + param.longflag
        elif param.flag:
            curFlag = '-' + param.flag
        else:
            continue

        if _is_on_girder(param) and param.name in kwargs['outputs']:

            curValue = os.path.join(
                '/data', hargs['params'][param.name + _girderOutputNameSuffix]
            )

            containerArgs.append(curFlag, curValue)


def _addReturnParameterFileToContainerArgs(containerArgs, kwargs, hargs):

    curName = _return_parameter_file_name

    if curName in kwargs['outputs']:

        curFlag = '--returnparameterfile'

        curValue = os.path.join(
            '/data', hargs['params'][curName + _girderOutputNameSuffix]
        )

        containerArgs.append(curFlag, curValue)


def _addIndexedParamsToContainerArgs(index_params, containerArgs, hargs):

    for param in index_params:

        if param.channel == 'input':

            if param.name in hargs:
                curValue = hargs[param.name]
            else:
                curValue = hargs['params'][param.name]

        elif param.channel == 'output':

            if not _is_on_girder(param):
                raise Exception(
                    'The type of indexed output parameter %d '
                    'must be of type - %s' % (
                        param.index,
                        _SLICER_TYPE_TO_GIRDER_MODEL_MAP.keys()
                    )
                )

            curValue = os.path.join(
                '/data', hargs['params'][param.name + _girderOutputNameSuffix]
            )

        else:
            continue

    containerArgs.append(curValue)


def genHandlerToRunDockerCLI(dockerImage, cliRelPath, restResource):
    """Generates a handler to run docker CLI using girder_worker

    Parameters
    ----------
    dockerImage : str
        Docker image in which the CLI resides
    cliRelPath : str
        Relative path of the CLI which is needed to run the CLI by running
        the command docker run `dockerImage` `cliRelPath`
    restResource : girder.api.rest.Resource
        The object of a class derived from girder.api.rest.Resource to which
        this handler will be attached

    Returns
    -------
    function
        Returns a function that runs the CLI using girder_worker
    """

    cliName = os.path.normpath(cliRelPath).replace(os.sep, '.')

    # get xml spec
    str_xml = subprocess.check_output(['docker', 'run', dockerImage,
                                       cliRelPath, '--xml'])

    # parse cli xml spec
    xmlFile = 'temp.xml'
    with open(xmlFile, 'w') as f:
        f.write(str_xml)

    clim = CLIModule(xmlFile)

    os.remove(xmlFile)

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
    taskSpec = {'name': cliName,
                'mode': 'docker',
                'docker_image': dockerImage,
                'inputs': [],
                'outputs': []}

    # get CLI parameters
    index_params, opt_params, simple_out_params = _getCLIParameters(clim)

    # add indexed input parameters
    index_input_params = filter(lambda p: p.channel == 'input', index_params)

    _addIndexedInputParams(index_input_params, taskSpec, handlerDesc)

    # add indexed output parameters
    index_output_params = filter(lambda p: p.channel == 'output', index_params)

    _addIndexedOutputParams(index_output_params, taskSpec, handlerDesc)

    # add optional input parameters
    opt_input_params = filter(lambda p: p.channel != 'output', opt_params)

    _addOptionalInputParams(opt_input_params, taskSpec, handlerDesc)

    # add optional output parameters
    opt_output_params = filter(lambda p: p.channel == 'output', opt_params)

    _addOptionalOutputParams(opt_output_params, taskSpec, handlerDesc)

    # add returnparameterfile if there are simple output params
    if len(simple_out_params) > 0:
        _addReturnParameterFileParam(taskSpec, handlerDesc)

    # define CLI handler function
    @boundHandler(restResource)
    @access.user
    @describeRoute(handlerDesc)
    def cliHandler(self, **hargs):

        user = self.getCurrentUser()
        token = self.getCurrentToken()['_id']

        # create job
        jobModel = self.model('job', 'jobs')
        jobTitle = '.'.join((restResource.resourceName, cliName))
        job = jobModel.createJob(title=jobTitle,
                                 type=jobTitle,
                                 handler='worker_handler',
                                 user=user)
        kwargs = {
            'validate': False,
            'auto_convert': True,
            'cleanup': True,
            'inputs': dict(),
            'outputs': dict()
        }

        # create job info
        jobToken = jobModel.createJobToken(job)
        kwargs['jobInfo'] = wutils.jobInfoSpec(job, jobToken)

        # create indexed input parameter bindings
        _addIndexedInputParamBindings(index_input_params,
                                      kwargs['inputs'], hargs, token)

        # create indexed output boundings
        _addIndexedOutputParamBindings(index_output_params,
                                       kwargs['outputs'], hargs, token)

        # create optional input parameter bindings
        _addOptionalInputParamBindings(opt_input_params,
                                       kwargs['inputs'], hargs, user, token)

        # create optional output parameter bindings
        _addOptionalOutputParamBindings(opt_output_params,
                                        kwargs['outputs'], hargs, user, token)

        # create returnparameterfile binding
        _addReturnParameterFileBinding(kwargs['outputs'], hargs, user, token)

        # construct container arguments
        containerArgs = [cliRelPath]

        _addOptionalInputParamsToContainerArgs(opt_input_params,
                                               containerArgs, hargs)

        _addOptionalOutputParamsToContainerArgs(opt_input_params,
                                                containerArgs, kwargs, hargs)

        _addReturnParameterFileToContainerArgs(containerArgs, kwargs, hargs)

        _addIndexedParamsToContainerArgs(index_params,
                                         containerArgs, hargs)

        # create task spec
        taskSpec['container_args'] = containerArgs
        kwargs['task'] = taskSpec

        # schedule job
        job['kwargs'] = kwargs
        job = jobModel.save(job)
        jobModel.scheduleJob(job)

        # return result
        return jobModel.filter(job, user)

    handlerFunc = cliHandler

    # loadmodel stuff for indexed input params on girder
    index_input_params_on_girder = filter(_is_on_girder, index_input_params)

    for param in index_input_params_on_girder:

        curModel = _SLICER_TYPE_TO_GIRDER_MODEL_MAP[param.typ]
        curMap = {param.name + _girderInputFileSuffix: param.name}

        handlerFunc = loadmodel(map=curMap,
                                model=curModel,
                                level=AccessType.READ)(handlerFunc)

    # loadmodel stuff for indexed output params on girder
    index_output_params_on_girder = filter(_is_on_girder, index_output_params)

    for param in index_output_params_on_girder:

        curModel = 'folder'
        curMap = {param.name + _girderOutputFolderSuffix: param.name}

        handlerFunc = loadmodel(map=curMap,
                                model=curModel,
                                level=AccessType.WRITE)(handlerFunc)

    return handlerFunc


def genHandlerToGetDockerCLIXmlSpec(dockerImage, cliRelPath, restResource):
    """Generates a handler that returns the XML spec of the docker CLI

    Parameters
    ----------
    dockerImage : str
        Docker image in which the CLI resides
    cliRelPath : str
        Relative path of the CLI which is needed to run the CLI by running
        the command docker run `dockerImage` `cliRelPath`
    restResource : girder.api.rest.Resource
        The object of a class derived from girder.api.rest.Resource to which
        this handler will be attached

    Returns
    -------
    function
        Returns a function that returns the xml spec of the CLI
    """

    str_xml = subprocess.check_output(['docker', 'run', dockerImage,
                                       cliRelPath, '--xml'])

    # define the handler that returns the CLI's xml spec
    @boundHandler(restResource)
    @access.user
    @describeRoute(
        Description('Get XML spec of %s CLI' % cliRelPath)
    )
    def getXMLSpecHandler(self, *args, **kwargs):
        return str_xml

    return getXMLSpecHandler


def genRESTEndPointsForSlicerCLIsInDocker(info, restResource, dockerImages):
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
    restResource : str or girder.api.rest.Resource
        REST resource to which the end-points should be attached
    dockerImages : str or list of str
        A single docker image or a list of docker images

    Returns
    -------

    """

    # validate restResource argument
    if not isinstance(restResource, (str, Resource)):
        raise Exception('restResource must either be a string or '
                        'an object of girder.api.rest.Resource')

    # validate dockerImages arguments
    if not isinstance(dockerImages, [str, list]):
        raise Exception('dockerImages must either be a single docker image '
                        'string or a list of docker image strings')

    if isinstance(dockerImages, list):
        for img in dockerImages:
            if not isinstance(img, str):
                raise Exception('dockerImages must either be a single '
                                'docker image string or a list of docker '
                                'image strings')
    else:
        dockerImages = [dockerImages]

    # create REST resource if given a name
    if isinstance(restResource, 'str'):
        restResource = type(restResource, (Resource, ), {})()

    # Add REST routes for slicer CLIs in each docker image
    cliList = []

    for dimg in dockerImages:

        # get CLI list
        cliListSpec = subprocess.check_output(['docker', 'run',
                                               dimg, '--list_cli'])

        # Add REST end-point for each CLI
        for cliRelPath in cliListSpec.keys():

            # create a POST REST route that runs the CLI
            try:
                cliRunHandler = genHandlerToRunDockerCLI(dimg,
                                                         cliRelPath,
                                                         restResource)
            except Exception as e:
                print "Failed to create REST endpoints for %s: %s" % (
                    cliRelPath, e)
                continue

            cliSuffix = os.path.normpath(cliRelPath).replace(os.sep, '_')

            cliRunHandlerName = 'run_' + cliSuffix
            setattr(restResource, cliRunHandlerName, cliRunHandler)
            restResource.route('POST',
                               (cliRelPath, 'run'),
                               getattr(restResource, cliRunHandlerName))

            # create GET REST route that returns the xml of the CLI
            try:
                cliGetXMLSpecHandler = genHandlerToGetDockerCLIXmlSpec(
                    dimg, cliRelPath, restResource)

            except Exception as e:
                print "Failed to create REST endpoints for %s: %s" % (
                    cliRelPath, e)
                continue

            cliGetXMLSpecHandlerName = 'get_xml_' + cliSuffix
            setattr(restResource,
                    cliGetXMLSpecHandlerName,
                    cliGetXMLSpecHandler)
            restResource.route('GET',
                               (cliRelPath, 'xmlspec',),
                               getattr(restResource, cliGetXMLSpecHandlerName))

            cliList.append(cliRelPath)

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
    setattr(info['apiRoot'], restResource.__name__, restResource)

    # return restResource
    return restResource
