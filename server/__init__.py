import os

from girder import events
from girder.utility.webroot import Webroot

from .rest_slicer_cli import genRESTEndPointsForSlicerCLIsInDocker
from .handlers import process_annotations
from .constants import *
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

    img_list = [str(key) for dic in getDockerImages() for key in list(dic)]
    print(img_list)
    print(getDockerImages())
    genRESTEndPointsForSlicerCLIsInDocker(
        info, 'HistomicsTK', img_list
        )


def getDockerImages():
    module_list = ModelImporter.model('setting').get(
        PluginSettings.DOCKER_IMAGES)
    if module_list is None:
        module_list = ModelImporter.model('setting').getDefault(
            PluginSettings.DOCKER_IMAGES)
    return module_list


events.bind('data.process', 'HistomicsTK', process_annotations)
events.bind('model.setting.validate',
            'histomicstk_modules', constants.validateSettings)
