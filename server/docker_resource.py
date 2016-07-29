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

from girder.api.v1.resource import Resource, RestException

from girder.utility.model_importer import ModelImporter

from girder.api import access
from girder.api.describe import Description, describeRoute
from .rest_slicer_cli import genRESTEndPointsForSlicerCLIsInDockerCache
from girder.plugins.jobs.constants import JobStatus


class DockerResource(Resource):
    """

    """

    resourceName = ''
    jobType = ''

    def __init__(self, name):
        super(DockerResource, self).__init__()
        self.currentEndpoints = {}
        self.resourceName = name
        self.jobType = name+'_job'
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
        .notes("""Must be a system administrator to call this. """)
        .errorResponse(
            'You are not a system administrator.', 403)
        .errorResponse(
            'Failed to set system setting.', 500)
    )
    def getDockerImages(self, params):
        dockermodel = ModelImporter.model('dockerimagemodel', 'HistomicsTK')
        dockerCache = dockermodel.loadAllImages()
        return dockerCache.getAllCliSpec()

    @access.admin
    @describeRoute(
        Description('Remove a docker image ')
        .notes(
            """Must be a system administrator to call this. """)
        .param('name', 'The name or a list of names  of the '
                       'docker images to be removed', required=True)
        .param('delete_from_local_repo',
               'Boolean True or False, if True the image is deleted from the'
               ' local repo, requiring it to be pulled from a repository the '
               'next time it is used. If False the meta data on the docker '
               'image is deleted but the docker image remains. This parameter '
               'is False by default', required=False)
        .errorResponse('You are not a system administrator.', 403)
        .errorResponse('Failed to set system setting.', 500)
    )
    # TODO delete REST endpoint
    def deleteImage(self, params):

        self.requireParams(('name',), params)
        name = params['name']
        if 'delete_from_local_repo' in params:
            deleteImage = params['delete_from_local_repo']
            if deleteImage is 'True':
                deleteImage = True
        else:
            deleteImage = False

        name = json.loads(name)
        nameList = []
        if isinstance(name, list):
            for img in name:

                if not isinstance(img, six.string_types):
                    raise RestException('%s was not a valid string.' % img)
            else:
                nameList = name
        elif isinstance(name, six.string_types):

            nameList = [name]

        else:

            raise RestException('name was not a valid JSON list or string.')

        self._deleteImage(nameList, deleteImage)

    def _deleteImage(self, names, deleteImage):
        """
        Removes the docker images and there respective clis from the settings
        :param name: The name of the docker image (user/rep:tag)

        """

        dockermodel = ModelImporter.model('dockerimagemodel', 'HistomicsTK')
        dockermodel.removeImages(names)

        self.deleteImageEndpoints(names)
        if deleteImage:

            dockermodel.delete_docker_image_from_repo(names)

    @access.admin
    @describeRoute(
        Description('Add a to the list of images to be loaded ').notes(
            """Must be a system administrator to call this.""").param(
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
        dockerimagemodel = ModelImporter.model('dockerimagemodel',
                                               'HistomicsTK')

        if isinstance(name, list):
            for img in name:

                if not isinstance(img, six.string_types):
                    raise RestException('%s was not a valid string.' % img)

        elif isinstance(name, six.string_types):
                name = [name]
        else:
            raise RestException('a valid string or a list of '
                                'strings was not passed in')
        dockerimagemodel.putDockerImage(name, self.jobType, True)

    def storeEndpoints(self, imgName, argList):
        if imgName in self.currentEndpoints:
            self.currentEndpoints[imgName].append(argList)
        else:
            self.currentEndpoints[imgName] = []
            self.currentEndpoints[imgName].append(argList)

    def deleteImageEndpoints(self, imageList=None):

        if imageList is None:
            imageList = self.currentEndpoints.keys()
        for imageName in imageList:
            if imageName in self.currentEndpoints:
                endpointList = self.currentEndpoints[imageName]
                for endpoint in endpointList:

                    self.removeRoute(endpoint[0], endpoint[1],
                                     getattr(self, endpoint[2]))
                    delattr(self, endpoint[2])
                del self.currentEndpoints[imageName]

    def AddRestEndpoints(self, event):
            job = event.info

            if job['type'] == self.jobType and job['status']\
                    == JobStatus.SUCCESS:

                    # remove all previos endpoints
                    dockermodel = ModelImporter.model('dockerimagemodel',
                                                      'HistomicsTK')
                    cache = dockermodel.loadAllImages()
                    # corner case where user manually ran docker rmi on an image
                    # that was loaded. Load all images will only return
                    # existing image but the old rest endpoint will still exist
                    self.deleteImageEndpoints()
                    genRESTEndPointsForSlicerCLIsInDockerCache(self, cache)
