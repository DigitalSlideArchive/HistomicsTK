# !/usr/bin/env python
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


import six
import json
from six import iteritems
import hashlib

from girder.api.v1.resource import Resource, RestException
from .constants import PluginSettings
from girder.utility.model_importer import ModelImporter
from girder.api.rest import getCurrentUser
from girder.api import access
from girder.api.describe import Description, describeRoute


class DockerResource(Resource):
    """
    Manages the exposed rest api. When the settings are updated te new list
    of docker images is checked, pre-loaded images will be ignored. New images
    will cause a job to generate the cli handler and generate the rest endpoint
    asynchronously.Deleted images will result in the removal of the rest api
    endpoint though docker will still cache the image unless removed manually
    (docker rmi image_name)
    """
    resourceName = ''

    def __init__(self, name):
        super(DockerResource, self).__init__()
        self.resourceName = name
        DockerResource.resourceName = name
        self.route('PUT', (DockerResource.resourceName, 'docker_image'),
                   self.setImages)
        self.route('DELETE', (DockerResource.resourceName, 'docker_image'),
                   self.deleteImage)
        self.route('GET', (DockerResource.resourceName, 'docker_image'),
                   self.getDockerImages)

    @access.admin
    @describeRoute(
        Description('list docker images and their clis ')
        .notes("""Must be a system administrator to call this. If the value
                       passed is a valid JSON object.""")
        .errorResponse(
            'You are not a system administrator.', 403)
        .errorResponse(
            'Failed to set system setting.', 500)
    )
    def getDockerImages(self, params):
        data = {}
        current = DockerCache(getDockerImageSettings())
        for image in current.getDockerImageList():
            data[image] = current.getCLIListSpec(image)
        return data

    @access.admin
    @describeRoute(
        Description('Remove a docker image ')
        .notes(
            """Must be a system administrator to call this. If the value
                       passed is a valid JSON object.""")
        .param('name', 'The name or a list of names  of the '
                       'docker images to be removed', required=True)
        .errorResponse('You are not a system administrator.', 403)
        .errorResponse('Failed to set system setting.', 500)
    )
    # TODO delete REST endpoint
    def deleteImage(self, params):

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
        """
        Removes the docker images and there respective clis from the settings
        :param name: The name of the docker image (user/rep:tag)

        """
        currentSettings = DockerCache(getDockerImageSettings())
        for name in names:

            if not currentSettings.deleteImage(name):
                raise RestException('%s does not exist' % name)
        # need to remove each cli attribute
        # need to remove the clli list attribute
        # need to remove the rest route
        print "after delete", currentSettings.getRawData()
        ModelImporter.model('setting').set(PluginSettings.DOCKER_IMAGES,
                                           currentSettings.getRawData())

    @access.admin
    @describeRoute(
        Description('Add a to the list of images to be loaded ').notes(
            """Must be a system administrator to call this. If the value
                   passed is a valid JSON object, it will be parsed and stored
                   as an object.""").param(
            'name', 'The name or a list of names  of the '
                    'docker images to be loaded ', required=True)
            .errorResponse('You are not a system administrator.', 403)
            .errorResponse('Failed to set system setting.', 500)
    )
    # TODO check the local cache and cloud for different images of same name
    # TODO how to handle newer images(take the latest or require a confirmation)
    # TODO use image id to confirm equivalence need v2 manifest schema on cloud
    # TODO how to handle duplicate clis
    # TODO create the new REST endpoints
    def setImages(self, params):
        """Validates the new images to be added (if they exist or not) and then
        attempts to collect xml data to be cached. a job is then called to
        update the PluginSettings.DOCKER_IMAGES settings with the new
        information
        """
        self.requireParams(('name',), params)
        name = params['name']
        name = json.loads(name)
        self.model('setting').set(PluginSettings.DOCKER_IMAGES, name)

    @staticmethod
    def appendImageJob(imgs):
        """
        Create an async job to check if each docker image in the list imgs exist
        For each existing image cli information will be cached and saved as
        a setting
        """
        currentSettings = DockerCache(getDockerImageSettings())

        job = ModelImporter.model('job', 'jobs').createLocalJob(
            module='girder.plugins.%s.'
                   'image_worker' % DockerResource.resourceName,
            function='loadXML',
            kwargs={'name': imgs,
                    'oldSettings': currentSettings.getRawData()},
            title='Updating Settings and Caching xml',
            type='%s.images' % DockerResource.resourceName,
            user=getCurrentUser(),
            public=True,
            async=True
        )
        ModelImporter.model('job', 'jobs').scheduleJob(job)

    @staticmethod
    def saveSettingsHook(event):
        key, val = event.info['key'], event.info['value']

        if key == PluginSettings.DOCKER_IMAGES:

            if isinstance(val, dict):
                # assume receiving data in the format of DockerCache
                # validate all data
                newSettings = DockerCache(val)
                try:
                    newSettings.validate()
                except DockerImageDataError as err:
                    event.defaultPrevented()
                    raise RestException(
                        'format of new settings was not correct %s', err)
            else:
                event.defaultPrevented()

    @staticmethod
    def validateSettings(event):
        """
        Takes three types of inputs, a single image name as a string, alist of
        images names or the raw format displayed below. If either an single
        docker image name or list of image names is provided cli information on
        those images are queried and stored as a setting in the raw format
        described below. If the raw format is provided, the structure
        and the validty of the contents are checked, if the raw format passes
        these check then it replaces the current settings value.
        Validity checks include whether
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
            currentSettings = DockerCache(getDockerImageSettings())
            if isinstance(val, dict):
                # assume receiving data in the format of DockerCache
                # validate all data
                newSettings = DockerCache(val)
                DockerResource.validateDict(newSettings)
            elif isinstance(val, list):
                # assume this is a list of desired images

                for img in val:

                    if not isinstance(img, six.string_types):
                        raise RestException('%s was not a valid string.' % img)
                else:
                    # do not want to save the list as the setting

                    DockerResource.appendImageJob(val)

            elif isinstance(val, six.string_types):
                # do not want to save the string as the setting

                DockerResource.appendImageJob([val])

            else:

                raise RestException('name was not a valid JSON list or string.')

            event.info['value'] = currentSettings.getRawData()
            event.preventDefault().stopPropagation()

    @staticmethod
    def validateDict(newSettings):
        currentSettings = DockerCache(getDockerImageSettings())
        job = ModelImporter.model('job', 'jobs').createLocalJob(
            module='girder.plugins.HistomicsTK.image_worker',
            function='verifyDictionary',
            kwargs={
                'newSettings': newSettings.getRawData(),
                'oldSettings': currentSettings.getRawData()
            },

            title='Updating Settings ',
            type='HistomicsTK.images',
            user=getCurrentUser(),
            public=True,
            async=True
        )
        ModelImporter.model('job', 'jobs').scheduleJob(job)


def getDockerImageSettings():
    module_list = ModelImporter.model('setting').get(
        PluginSettings.DOCKER_IMAGES)
    if module_list is None:
        module_list = {}
    return module_list


class DockerCache():

    imageName = 'docker_image_name'
    type = 'type'
    xml = 'xml'
    cli_dict = 'cli_list'

    def __init__(self, cache):
        """
        Data is stored in the following format:
            {image_name_hash:
                {
                cli_list:{
                    cli_name:{
                        type:<type>
                        xml:<xml>

                            }
                        }
                docker_image_name:<name>
                }
            }
        """
        if isinstance(cache, dict):
            self.data = cache
        elif isinstance(cache, str):
            if cache != '':
                self.data = json.loads(cache)
        else:
            self.data = {}

    def getDockerImageList(self):

        return [str(imgDict[DockerCache.imageName])
                for (imgHash, imgDict) in iteritems(self.data)]

    def isImageAlreadyLoaded(self, name):
        imageKey = self._getHashKey(name)
        if imageKey in self.data:
            return True
        else:
            return False

    def addImage(self, name):

        if not isinstance(name, six.string_types):
            raise DockerImageError('Image name should be a string,'
                                   ' could not add the image', 'bad name')

        imageKey = self._getHashKey(name)
        self.data[imageKey] = {}

        self.data[imageKey][DockerCache.imageName] = name
        self.data[imageKey][DockerCache.cli_dict] = {}

    def addCLI(self, img_name, cli_name, type, xml):

        cliData = {}
        cliData[DockerCache.type] = type
        cliData[DockerCache.xml] = xml

        imageKey = self._getHashKey(img_name)
        self.data[imageKey][DockerCache.cli_dict][cli_name] = cliData

    def getRawData(self):
        return self.data

    def deleteImage(self, name):
        imageKey = self._getHashKey(name)
        if imageKey in self.data:
            del self.data[imageKey]
            return True
        else:
            return False

    def _getHashKey(self, imgName):
        imageKey = hashlib.sha256(imgName.encode()).hexdigest()
        return imageKey

    def getCLIXML(self, imgName, cli):
        imgKey = self._getHashKey(imgName)
        if imgKey in self.data:
            imageData = self.data[imgKey]

        else:
            print('no image named %s' % imgName)
            return None

        if cli in imageData[DockerCache.cli_dict]:

            return imageData[DockerCache.cli_dict][cli][DockerCache.xml]
        else:
            print('no cli named %s' % imgName)
            return None

    def getCLIListSpec(self, imgName):
        cliDict = {}
        imgKey = self._getHashKey(imgName)
        if imgKey in self.data:
            imgData = self.data[imgKey]
            for (key, val) in iteritems(imgData[DockerCache.cli_dict]):
                cliDict[key] = {DockerCache.type: val[DockerCache.type]}
            return cliDict
        else:
            return None

    def validate(self):
        """
        Enforce structure of the data, does not verify that xml and cli
        types field values are appropriate
        """
        if self.data is {}:
            return True
        for (imgHash, imgData) in iteritems(self.data):
            if DockerCache.imageName not in imgData:
                raise DockerImageDataError('There is no key: '
                                           '%s in the dict'
                                           % DockerCache.imageName)
            if imgHash != self._getHashKey(imgData[DockerCache.imageName]):
                raise DockerImageDataError('The hash id is not the sha256 '
                                           'hash of the name: %' %
                                           imgData[DockerCache.imageName])
            if DockerCache.cli_dict not in imgData:
                raise DockerImageDataError('There is no dictionary of cli ')
            for (cliName, cliData) in iteritems(imgData[DockerCache.cli_dict]):
                if DockerCache.type not in cliData:
                    raise DockerImageDataError('The cli %s does not have a key'
                                               ' for the type of cli' % cliName)
                if DockerCache.xml not in cliData:
                    raise DockerImageDataError('The cli %s does not have a key '
                                               'for the xml of cli' % cliName)

    def equals(self, cache):
        if not isinstance(cache, DockerCache):
            raise DockerImageError(" Can only compare a Dockercache "
                                   "object to another DockerCache object")
        return self.data == cache.data


class DockerImageError(Exception):
    def __init__(self, message, image_name):
        self.message = message
        self.imageName = image_name
        super(DockerImageError, self).__init__(self, message)


class DockerImageNotFoundError (DockerImageError):
        def __init__(self, message, image_name, locations=[]):
            super(DockerImageNotFoundError, self).__init__(message, image_name)
            # list of registries tried(local dockerhub etc )
            self.locations = locations


class DockerImageDataError(DockerImageError):
    def __init__(self, message, image_name):
        super(DockerImageDataError, self).__init__(message, image_name)
