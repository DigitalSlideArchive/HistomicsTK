import os

from girder import events
from girder.utility.webroot import Webroot

from .rest_slicer_cli import genRESTEndPointsForSlicerCLIsInDockerCache
from .handlers import process_annotations
from .constants import PluginSettings
from .docker_resource import DockerResource, DockerCache
from girder.utility.model_importer import ModelImporter
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

    dockerCache = DockerCache(docker_resource.getDockerImageSettings())

    genRESTEndPointsForSlicerCLIsInDockerCache(info, resource, dockerCache)

    events.bind('data.process', 'HistomicsTK', process_annotations)
    events.bind('model.setting.validate', 'histomicstk_modules',
                DockerResource.validateSettings)
    # events.bind('model.setting.save','histomicstk_modules',docker_resource.saveSettingsHook)
