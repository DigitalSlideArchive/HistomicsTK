
import six
import subprocess

from girder.constants import SettingDefault
from girder.models.model_base import ValidationException


class PluginSettings(object):

    DOCKER_IMAGES = 'histomicstk.docker_images'


# TODO check locally and on the cloud for image
# TODO check if a newer version should be pulled
def properImageName(image):
    return True


def validateSettings(event):
    """Check whether the new list of images or new image is valid.
        Validity checks include whether the image is already loaded
        (already has an exposed rest endpoint) and whether the image(s) are
        actually images either within the docker cache locally or on an
        available registry (default registry being dockerhub)
        """
    # val should be a list of dictionaries
    key, val = event.info['key'], event.info['value']

    if key == PluginSettings.DOCKER_IMAGES:
        print(type(val))
        if isinstance(val, dict):
            properImageName(val)
        elif isinstance(val, list):

            for data in val:
                print(data)
                if isinstance(data, dict):
                    properImageName(data)

        else:
            print(val)
            raise ValidationException(
                'Docker images were not proper')

        event.preventDefault().stopPropagation()


def localDockerImageExists(imageName):
    """checks the local docker cache for the image
    :param imageName: the docker image name in the form of repo/name:tag
    if the tag is not given docker defaults to using the :latest tag
    :type imageName: string
    :returns: if the image exit the id(sha256 hash) is returned otherwise
    None is returned
    """
    try:
        # docker inspect returns non zero if the image is not available
        # locally
        data = subprocess.checkoutput('docker', 'inspect',
                                      '--format="{{json .Id}}"', imageName)
    except subprocess.CalledProcessError as err:
        # the image does not exist locally, try to pull from dockerhub

        return None

    return data


SettingDefault.defaults[PluginSettings.DOCKER_IMAGES] = \
    {'dsarchive/histomicstk:v0-1-3': None}
