import os

from girder import events
from girder.utility.webroot import Webroot

from girder.plugins.slicer_cli_web.rest_slicer_cli import (
    genRESTEndPointsForSlicerCLIsInDockerCache
)

from girder.plugins.slicer_cli_web.docker_resource import DockerResource

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

    # create root resource for all REST end points of HistomicsTK
    resource = DockerResource('HistomicsTK')
    setattr(info['apiRoot'], resource.resourceName, resource)

    # load docker images from cache
    dockerImageModel = ModelImporter.model('docker_image_model',
                                           'slicer_cli_web')
    dockerCache = dockerImageModel.loadAllImages()

    # generate REST end points for slicer CLIs of each docker image
    genRESTEndPointsForSlicerCLIsInDockerCache(resource, dockerCache)

    # auto-ingest annotations into database when a .anot file is uploaded
    events.bind('data.process', 'HistomicsTK', process_annotations)

    events.bind('jobs.job.update.after', resource.resourceName,
                resource.AddRestEndpoints)
