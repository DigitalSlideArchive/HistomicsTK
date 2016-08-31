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
from models import DockerImageNotFoundError, DockerImage


# TODO add restpoint information in the get endpoint
class DockerResource(Resource):
    """
    Resource object that handles runtime generation and deletion of rest
    endpoints
    """

    resourceName = ''
    jobType = 'DockerImageDataJob'

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
        cache = dockerCache.getImages()
        data = {}
        for val in cache:
            name, tag, imgData = self.createRestDataForImageVersion(val)
            data.setdefault(name, {})[tag] = imgData

        return data

    def createRestDataForImageVersion(self, dockerImage):
        """
        Creates a dictionary with rest endpoint information for the given
        DockerImage object

        :param dockerImage: DockerImage object

        Returns: structured dictionary documentin clis and rest
        endpoints for this image version

        """

        name = dockerImage.name
        endpointData = self.currentEndpoints[name]

        if ':' in name:
            imageAndTag = name.split(':')
        else:
            imageAndTag = name.split('@')
        userAndRepo = imageAndTag[0]
        tag = imageAndTag[1]

        data = {}
        cli_dict = dockerImage.getCLIListSpec()

        for (cli, val) in six.iteritems(cli_dict):
            data[cli] = {}

            data[cli][DockerImage.type] = val
            cli_endpoints = endpointData[cli]

            for (operation, endpointRoute) in six.iteritems(cli_endpoints):
                cli_list = endpointRoute[1]
                if cli in cli_list:

                    data[cli][operation] = '/'+self.resourceName + \
                                           '/' + '/'.join(cli_list)
        return userAndRepo, tag, data

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
        Removes the docker images and there respective clis endpoints
        :param name: The name of the docker image (user/rep:tag)
        :type name: string
        :param deleteImage: Boolean indicating whether to delete the docker
        image from the local machine.(if True this is equivalent to
        docker rmi -f <image> )
        """

        dockermodel = ModelImporter.model('dockerimagemodel', 'HistomicsTK')
        try:

            dockermodel.removeImages(names)

            self.deleteImageEndpoints(names)
            if deleteImage:

                dockermodel.delete_docker_image_from_repo(names, self.jobType)
        except DockerImageNotFoundError as err:
            raise RestException('Invalid docker image name. '+str(err))

    @access.admin
    @describeRoute(
        Description('Add a to the list of images to be loaded ').notes(
            """Must be a system administrator to call this.""").param(
            'name', 'The name or a list of names  of the '
                    'docker images to be loaded ', required=True)
            .errorResponse('You are not a system administrator.', 403)
            .errorResponse('Failed to set system setting.', 500)
    )
    def setImages(self, params):
        """Validates the new images to be added (if they exist or not) and then
        attempts to collect xml data to be cached. a job is then called to
        update the girder collection containing the cached docker image data
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
                if ':' not in img or '@' not in img:
                    raise RestException('Image %s does not have a tag or '
                                        'digest' % img)

        elif isinstance(name, six.string_types):
                name = [name]
        else:
            raise RestException('a valid string or a list of '
                                'strings was not passed in')
        dockerimagemodel.putDockerImage(name, self.jobType, True)

    def storeEndpoints(self, imgName, cli, operation, argList):
        """
        information on each rest endpoint is saved so they can be
        deleted when docker images are removed or loaded
        :param imgName: The full name of the docker image with the tag.
        This name must match exactly with the name the command
        docker images displays in the console
        :type imgName: string
        :param cli: The name of the cli whose rest endpoint is being stored. The
        cli must match exactly with what teh docker image returns when
        running <docker image> --list_cli
        :type cli: string
        :param operation: The action the rest endpoint will execute run or
        xmlspec
        :type operation: string
        :argList:list of details for a specific endpoint. The arglist should
        contain [method,route_tuple,endpoint_method_handler_name].The route
        tuple should consist of the docker image name, the cli name and the
        operation. Since this tuple forms the rest route, the exact docker
        image name may not be used due to ':' used in
        docker image names. As a result the docker name and cli name used
        in the rest route do not have to match the actual docker name and cli
        name

        """
        if imgName in self.currentEndpoints:
            if cli in self.currentEndpoints[imgName]:
                self.currentEndpoints[imgName][cli][operation] = argList
            else:
                self.currentEndpoints[imgName][cli] = {}
                self.currentEndpoints[imgName][cli][operation] = argList

        else:
            self.currentEndpoints[imgName] = {}
            self.currentEndpoints[imgName][cli] = {}
            self.currentEndpoints[imgName][cli][operation] = argList

    def deleteImageEndpoints(self, imageList=None):

        if imageList is None:
            imageList = self.currentEndpoints.keys()
        for imageName in imageList:
            if imageName in self.currentEndpoints:
                for (cli, val) in six.iteritems(
                        self.currentEndpoints[imageName]):
                    for (operation, endpoint) in six.iteritems(val):
                        try:
                            self.removeRoute(endpoint[0], endpoint[1],
                                             getattr(self, endpoint[2]))
                            delattr(self, endpoint[2])
                        except Exception as err:
                            print err
            del self.currentEndpoints[imageName]

    def AddRestEndpoints(self, event):
        """
        Determines if the job event being triggered is due to the caching of
        new docker images or deleting a docker image off the local machine
        .If a new image is being loaded all old rest endpoints are deleted
        and endpoints fro all cached docker images are regenerated
        :param event: An event dictionary
        """

        job = event.info['job']
        status = event.info['params'].get('status', job['status'])

        if job['type'] == self.jobType and status == JobStatus.SUCCESS:
            # remove all previous endpoints
            dockermodel = ModelImporter.model('dockerimagemodel',
                                              'HistomicsTK')
            cache = dockermodel.loadAllImages()

            self.deleteImageEndpoints()
            genRESTEndPointsForSlicerCLIsInDockerCache(self, cache)
