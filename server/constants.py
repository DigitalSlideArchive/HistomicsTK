
import six
import subprocess

from girder.constants import SettingDefault
from girder.models.model_base import ValidationException


class PluginSettings(object):

    DOCKER_IMAGES = 'histomicstk.docker_images'

SettingDefault.defaults[PluginSettings.DOCKER_IMAGES] = \
    [{}, 'dsarchive/histomicstk:v0.1.3']
