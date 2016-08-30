import os

from girder import events
from girder.utility.webroot import Webroot

from girder.plugins.slicer_cli_web.rest_slicer_cli import (
    genRESTEndPointsForSlicerCLIsInDockerCache
)

from girder.plugins.slicer_cli.docker_resource import DockerResource

from .handlers import process_annotations

from girder.models.model_base import ModelImporter
_template = os.path.join(
    os.path.dirname(__file__),
    'webroot.mako'
)


def load(info):

    girderRoot = info['serverRoot']
    histomicsRoot = Webroot(_template)
    histomicsRoot.updateHtmlVars(girderRoot.vars)
    histomicsRoot.updateHtmlVars({'title': 'HistomicsTK'})

    info['serverRoot'].histomicstk = histomicsRoot
    info['serverRoot'].girder = girderRoot

    # passed in resource name must match the attribute added to info[apiroot]
    resource = DockerResource('HistomicsTK')
    info['apiRoot'].HistomicsTK = resource

    dockerImageModel = ModelImporter.model('docker_image_model',
                                           'slicer_cli_web')
    dockerCache = dockerImageModel.loadAllImages()

    genRESTEndPointsForSlicerCLIsInDockerCache(resource, dockerCache)

    events.bind('data.process', 'HistomicsTK', process_annotations)

    events.bind('model.job.save.after', resource.resourceName,
                resource.AddRestEndpoints)