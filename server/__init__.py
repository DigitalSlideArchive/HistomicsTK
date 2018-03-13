import json
import os

from girder import events
from girder.api import access
from girder.api.describe import Description, describeRoute
from girder.exceptions import ValidationException
from girder.utility.webroot import Webroot
from girder.utility import setting_utilities

from girder.plugins.slicer_cli_web.rest_slicer_cli import (
    genRESTEndPointsForSlicerCLIsInDockerCache
)

from girder.plugins.slicer_cli_web.docker_resource import DockerResource

from .handlers import process_annotations
import constants

from girder.models.model_base import ModelImporter
_template = os.path.join(
    os.path.dirname(__file__),
    'webroot.mako'
)


@setting_utilities.validator({
    constants.PluginSettings.HISTOMICSTK_DEFAULT_DRAW_STYLES
})
def validateListOrJSON(doc):
    val = doc['value']
    try:
        if isinstance(val, list):
            doc['value'] = json.dumps(val)
        elif val is None or val.strip() == '':
            doc['value'] = None
        else:
            parsed = json.loads(val)
            if not isinstance(parsed, list):
                raise ValueError
            doc['value'] = val.strip()
    except (ValueError, AttributeError):
        raise ValidationException('%s must be a JSON list.' % doc['key'], 'value')


class HistomicsTKResource(DockerResource):
    def __init__(self, name, *args, **kwargs):
        super(HistomicsTKResource, self).__init__(name, *args, **kwargs)
        self.route('GET', ('settings',), self.getPublicSettings)

    @describeRoute(
        Description('Get public settings for HistomicsTK.')
    )
    @access.public
    def getPublicSettings(self, params):
        keys = [
            constants.PluginSettings.HISTOMICSTK_DEFAULT_DRAW_STYLES,
        ]
        return {k: self.model('setting').get(k) for k in keys}


def _saveJob(event):
    """
    When a job is saved, if it is a docker run task, add the Dask Bokeh port to
    the list of exposed ports.
    """
    job = event.info
    try:
        from bson import json_util

        jobkwargs = json_util.loads(job['kwargs'])
        if ('docker_run_args' not in jobkwargs['task'] and
                'scheduler_address' in jobkwargs['inputs']):
            jobkwargs['task']['docker_run_args'] = {'ports': {'8787': None}}
            job['kwargs'] = json_util.dumps(jobkwargs)
    except Exception:
        pass


def load(info):

    girderRoot = info['serverRoot']
    histomicsRoot = Webroot(_template)
    histomicsRoot.updateHtmlVars(girderRoot.vars)
    histomicsRoot.updateHtmlVars({'title': 'HistomicsTK'})

    info['serverRoot'].histomicstk = histomicsRoot
    info['serverRoot'].girder = girderRoot

    pluginName = 'HistomicsTK'
    # create root resource for all REST end points of HistomicsTK
    resource = HistomicsTKResource(pluginName)
    setattr(info['apiRoot'], resource.resourceName, resource)

    # load docker images from cache
    dockerImageModel = ModelImporter.model('docker_image_model',
                                           'slicer_cli_web')
    dockerCache = dockerImageModel.loadAllImages()

    # generate REST end points for slicer CLIs of each docker image
    genRESTEndPointsForSlicerCLIsInDockerCache(resource, dockerCache)

    # auto-ingest annotations into database when a .anot file is uploaded
    events.bind('data.process', pluginName, process_annotations)

    events.bind('jobs.job.update.after', resource.resourceName,
                resource.AddRestEndpoints)

    events.bind('model.job.save', pluginName, _saveJob)
