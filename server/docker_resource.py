
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
import six
import json
from six import iteritems
import subprocess
import hashlib
from girder.api.v1.resource import Resource,RestException
from .constants import PluginSettings
from girder.models.model_base import ValidationException
from girder.utility.model_importer import ModelImporter
from girder.api.rest import boundHandler,getCurrentUser
from girder.api import access
from girder.api.describe import Description, describeRoute


class DockerResource(Resource):
    """Manages the exposed rest api. When the settings are updated te new list
    of docker images is checked, pre-loaded images will be ignored. New images will
    cause a job to generate the cli handler and generate the rest endpoint
    asynchronously.Deleted images will result in the removal of the rest api
    endpoint though docker will still cache the image unless removed manually
    (docker rmi image_name)
    """


    def __init__(self):
        super(DockerResource, self).__init__()
        self.resourceName = 'HistomicsTK'
        self.route('PUT', ('add_docker_image',), self.appendImage)
        self.route('DELETE', ('docker_image',), self.deleteImage)

    @access.admin
    @describeRoute(
        Description('Remove a docker image from histomicstk')
            .notes("""Must be a system administrator to call this. If the value
                       passed is a valid JSON object.""")
            .param('name', 'The name or a list of names  of the '
                           'docker images to be removed', required=True)
            .errorResponse('You are not a system administrator.', 403)
            .errorResponse('Failed to set system setting.', 500)
    )
    # TODO delete cli instance if they exist
    def deleteImage(self, params):
        """Removes the docker images and there respective clis from
        histomicstk
        """
        self.requireParams(('name',), params)
        name = params['name']
        name = json.loads(name)

        if isinstance(name, list):
            for img in name:

                if not isinstance(img, six.string_types):
                    raise RestException('%s was not a valid string.' % img)
            else:
                self._deleteImage(name)
        elif isinstance(name, six.string_types):

            self._deleteImage([name])

        else:

            raise RestException('name was not a valid JSON list or string.')

    def _deleteImage(self, names):
        currentSettings = DockerCache(getDockerImages())
        for name in names:

            if not currentSettings.deleteImage(name):
                raise RestException('%s does not exist' % name)
        ModelImporter.model('setting').set(PluginSettings.DOCKER_IMAGES,
                                           currentSettings.raw())

    @access.admin
    @describeRoute(
        Description('Add a to the list of images to be loaded by histomics tk')
            .notes("""Must be a system administrator to call this. If the value
                   passed is a valid JSON object, it will be parsed and stored
                   as an object.""")
            .param('name', 'The name or a list of names  of the '
                           'docker images to be loaded ', required=True)
            .errorResponse('You are not a system administrator.', 403)
            .errorResponse('Failed to set system setting.', 500)
        )
    # TODO check the local cache and cloud for different images of same name
    # TODO how to handle newer images(take the latest or require a confirmation)
    # TODO use image id to confirm equivalence need v2 manifest schema on cloud
    # TODO how to handle duplicate clis
    # TODO have an event update (create cli instances)
    # TODO check if the new images exist
    def appendImage(self, params):
        """Validates the new images to be added (if they exist or not) and then
        attempts to collect xml data to be cached. a job is then called to update
        the histomicstk.docker_images settings with the new information
        """
        self.requireParams(('name',), params)
        name = params['name']
        name = json.loads(name)

        if isinstance(name, list):
            for img in name:

                if not isinstance(img, six.string_types):

                    raise RestException('%s was not a valid string.' % img)
            else:
                self._appendImage(name)
        elif isinstance(name, six.string_types):

            self._appendImage([name])

        else:

            raise RestException('name was not a valid JSON list or string.')

    def _appendImage(self, imgs):
        currentSettings = DockerCache(getDockerImages())
        for img in imgs:
            if currentSettings.imageAlreadyLoaded(img):
                raise RestException('image %s already is loaded.'% img)

        job = ModelImporter.model('job', 'jobs').createLocalJob(
            module='girder.plugins.HistomicsTK.image_worker',
            function='loadXML',
            kwargs={'name': imgs
                    },
            title='Updating Settings and Caching xml',
            type='HistomicsTK.images',
            user=getCurrentUser(),
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
            {image_name_hash:
                {
                cli_name:{
                    type:
                    xml:

                    }
                docker_image_name:name
                }
            }
        """

    # val should be a dictionary of dictionaries
    key, val = event.info['key'], event.info['value']

    if key == PluginSettings.DOCKER_IMAGES:

        cachedData = None
        if isinstance(val, dict):

            for (dictKey, dictValue) in iteritems(val):
                pass
                # checkOldImage(dictKey, dictValue)
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
        module_list = {}
    return module_list


class DockerCache() :
    imageName='docker_image_name'
    type='type'
    xml='xml'

    def __init__(self, cache):
        """
        Data is stored in the following format:
            {image_name_hash:
                {
                cli_name:{
                    type:
                    xml:

                    }
                docker_image_name:name
                }
            }
        """

        self.data = cache

    def getDockerImg(self):
        nameList = []
        for (imgHash, imgDict) in iteritems(self.data):
            nameList.append(str(imgDict[self.imageName]))

        return nameList

    def imageAlreadyLoaded(self, name):
        imageKey = self._getHashKey(name)
        if imageKey in self.data:
            return True
        else:
            return False

    def addImage(self, name):
        imageKey = self._getHashKey(name)
        self.data[imageKey] = {}

        self.data[imageKey][self.imageName] = name

    def addCLI(self, img_name, cli_name, type, xml):
        cli = {}
        cliData = {}
        cliData[self.type] = type
        cliData[self.xml] = xml

        imageKey = self._getHashKey(img_name)
        self.data[imageKey][cli_name] = cliData

    def raw(self):
        return self.data

    def deleteImage(self, name):
        imageKey = self._getHashKey(name)
        if imageKey in self.data:
            del self.data[imageKey]
            return True
        else:
            return False
    def _getHashKey(self,imgName):
        imageKey = hashlib.sha256(imgName.encode()).hexdigest()
        return imageKey

    def getCLIXML(self,imgName,cli):
        imgKey = self._getHashKey(imgName)
        if imgKey in self.data:
            imageData=self.data[imgKey]
        else:
            return None

        if cli in imageData:
            return imageData[self.xml]
        else:

            return None