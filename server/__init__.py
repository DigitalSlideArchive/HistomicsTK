import os

from girder import events
from girder.utility.webroot import Webroot

from .rest_slicer_cli import genRESTEndPointsForSlicerCLIsInDocker
from .handlers import process_annotations
from .constants import PluginSettings
from .docker_resource import validateSettings
from girder.utility.model_importer import ModelImporter
_template = os.path.join(
    os.path.dirname(__file__),
    'webroot.mako'
)


def load(info):
    images = ModelImporter.model('setting').getDefault(
            constants.PluginSettings.DOCKER_IMAGES)
    ModelImporter.model('setting').set(PluginSettings.DOCKER_IMAGES, images)

    girderRoot = info['serverRoot']
    histomicsRoot = Webroot(_template)
    histomicsRoot.updateHtmlVars(girderRoot.vars)
    histomicsRoot.updateHtmlVars({'title': 'HistomicsTK'})

    info['serverRoot'].histomicstk = histomicsRoot
    info['serverRoot'].girder = girderRoot


    #genRESTEndPointsForSlicerCLIsInDocker(info, 'HistomicsTK', img_list)

events.bind('data.process', 'HistomicsTK', process_annotations)
events.bind('model.setting.validate','histomicstk_modules', docker_resource.validateSettings)
events.bind('model.setting.save.after', 'histomicstk_modules_save',docker_resource.saveSettings)
