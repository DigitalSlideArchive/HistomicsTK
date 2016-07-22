
#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
#  Copyright Kitware Inc.
#
#  Licensed under the Apache License, Version 2.0 ( the "License" );
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

import os

from six import iteritems
import subprocess
from girder.api.v1.resource import Resource
from .constants import PluginSettings
from girder.models.model_base import ValidationException
from girder.utility.model_importer import ModelImporter
from girder.api.rest import boundHandler
from girder.api import access


class DockerResource(Resource):
    """Manages the exposed rest api. When the settings are updated te new list
    of docker images is checked, pre-loaded images will be ignored. New images will
    cause a job to generate the cli handler and generate the rest endpoint
    asynchronously.Deleted images will result in the removal of the rest api
    endpoint though docker will still cache the image unless removed manually
    (docker rmi image_name)
    """
    loadedModules = []


def saveSettings(event):

    job = ModelImporter.model('job', 'jobs').createLocalJob(
        module='girder.plugins.HistomicsTK.image_worker',
        function='loadXML',
        kwargs={

        },
        title='Updating Settings and Caching xml',
        type='HistomicsTK.images',
        user=None,
        public=True,
        async=True
    )
    ModelImporter.model('job', 'jobs').scheduleJob(job)


# TODO remove bad image names
def validateSettings(event):
    """Check whether the new list of a single dictionary  is valid.
        Validity checks include whether the image is already loaded
        (already has an exposed rest endpoint) and whether the image(s) are
        actually images either within the docker cache locally or on an
        available registry (default registry being dockerhub)

        Data is stored in the following format:
           [ {image_name_hash:
                    {id:image:id
                    name:image_name,
                    xml:cli_spec
                    },
                    }]
        Newly added images will be stored as follows:
            [{},"new_image_name1","new_image_name2"]
        """

    # val should be a dictionary of dictionaries
    key, val = event.info['key'], event.info['value']

    if key == PluginSettings.DOCKER_IMAGES:
        print(type(val))
        cachedData = None
        if isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    cachedData = item
                    for (dictKey, dictValue) in iteritems(item):
                        pass
                        # checkOldImage(dictKey, dictValue)
                else:
                    pass
                    # checkNewImage(item, cachedData)

        else:

            print("not proper")
            raise ValidationException('Docker images were not proper')

        event.preventDefault().stopPropagation()


# TODO check local docker cache and default registry
def imageExists(name):
    """Given an image name determine if the image exists locally or on the default
    registry"""

    return True


def checkOldImage(key, dataDict):
    localImage = localDockerImageExists(dataDict["name"])
    if localImage and localImage != key:
        print('docker cache does not match the version in girder')
        print ('compare keys:\n,%s\n,%s' % key, localImage)


def checkNewImage(name,cache):
    foundImage = False
    for (dictKey, dictValue) in iteritems(cache):
        if dictValue and "name" in dictValue:
            if name == dictValue["name"]:
                foundImage=True
    if foundImage or not imageExists(name):
        return False
    else:
        return True


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


def getDockerImages():
    module_list = ModelImporter.model('setting').get(
        PluginSettings.DOCKER_IMAGES)
    if module_list is None:
        module_list = ModelImporter.model('setting').getDefault(
            PluginSettings.DOCKER_IMAGES)
    return module_list
