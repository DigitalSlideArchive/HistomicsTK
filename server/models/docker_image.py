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


from six import iteritems, string_types
import hashlib
import jsonschema


class DockerImageError(Exception):
    def __init__(self, message, image_name='None'):

        self.message = message
        # can be a string or list
        self.imageName = image_name
        Exception.__init__(self, message)

    def __str__(self):
        if isinstance(self.imageName, list):
            return self.message + ' (image names ' \
                                  '[' + ','.join(self.imageName)+']'
        elif isinstance(self.imageName, string_types):
            return self.message + ' (image name: ' + self.imageName+' )'
        else:
            return self.message


class DockerImageNotFoundError(DockerImageError):
    def __init__(self, message, image_name, locations=[]):
        super(DockerImageNotFoundError, self).__init__(message, image_name)
        # list of registries tried(local dockerhub etc )
        self.locations = locations


class DockerImage():
    """
    Represents docker image and contains metadata on a specific image
    """
    # keys used by the dictionary that stores metadata on the image
    imageName = 'docker_image_name'
    imageHash = 'imagehash'
    type = 'type'
    xml = 'xml'
    cli_dict = 'cli_list'
    # structure of the dictionary to store meta data
    # {
    # imagehash:<hash of docker image name>
    #     cli_list: {
    #         cli_name: {
    #                   type: < type >
    #                   xml: < xml >
    #
    # }
    # }
    # docker_image_name: < name >
    # }

    def __init__(self, name):
        try:
            if isinstance(name, string_types):

                imageKey = DockerImage.getHashKey(name)
                self.data = {}
                self.data[DockerImage.imageName] = name
                self.data[DockerImage.cli_dict] = {}
                self.data[DockerImage.imageHash] = imageKey
                self.hash = imageKey
                self.name = name
                # TODO check/validate schema of dict
            elif isinstance(name, dict):
                jsonschema.validate(name, DockerImageStructure.ImageSchema)
                self.data = name.copy()
                self.name = self.data[DockerImage.imageName]
                self.hash = DockerImage.getHashKey(self.name)
            else:

                raise DockerImageError('Image should be a string, or dict'
                                       ' could not add the image',
                                       'bad init val')
        except Exception as err:
                raise DockerImageError('Could not initialize instance'
                                       ' of Docker Image \n'+str(err))

    def addCLI(self, cli_name, cli_data):
        """
        Add metadata on a specific cli
        :param cli_name: the name of the cli
        :param cli_data: a dictionary following the format:
                    {
                      type: < type >
                      xml: < xml >

                    }
        The data is passed in a s a dictionary in the case the more metadata
        is added to eh cli description
        """
        self.data[DockerImage.cli_dict][cli_name] = cli_data

    @staticmethod
    def getHashKey(imgName):
        """
        Generates a hash key (on the docker image name) used by the DockerImage
         object to provide a means to uniquely find the image meta data
         in the girder-mongo database. This prevents user defined image name
         from causing issues with pymongo.Note this key is not the same as the
         docker image id that the docker engine generates
        :imgName: The name of the docker image

        :returns: The hashkey as a string
        """
        imageKey = hashlib.sha256(imgName.encode()).hexdigest()
        return imageKey

    def getCLIXML(self, cli):

        if cli in self.data[DockerImage.cli_dict]:
            return self.data[DockerImage.cli_dict][cli][DockerImage.xml]

        else:
            raise DockerImageError('No cli named %s in the '
                                   'image %s' % (cli, self.name))

    def getCLIListSpec(self):
        """
        Returns a dictionary in the format of slicer_cli_list.json
        {
          <cli_name>: {
            "type"    : <algorithm format(python or R)>
          },
          <cli_name>: {
            "type"    : <algorithm format(python or R)>
          }
        }
        """
        spec_dict = {}
        for (key, val) in iteritems(self.data[DockerImage.cli_dict]):
            spec_dict[key] = val[DockerImage.type]
        return spec_dict

    def getRawData(self):
        return self.data


class DockerCache:
    """
    This class is used to hold and access meta data on existing images
    """
    def __init__(self):
        """
        Data is stored in the following format:
            {image_name_hash:DockerImage

            }
        """

        self.data = {}

    def addImage(self, img):
        """
        Add an image object to the Docker cache
        :param img: A docker image object
        :type img: DockerImage
        """
        try:
            if isinstance(img, DockerImage):
                self.data[img.hash] = DockerImage(img.getRawData())
            else:
                raise DockerImageError('Tried to add a non '
                                       'docker image object to cache')
        except Exception as err:
            raise DockerImageError('Failed to add the '
                                   'docker image to the cache' + str(err))

    def getImageNames(self):
        """
        Get the list docker image names in the cache
        """
        return [img.name for img in self.data.values()]

    def getImages(self):
        """
        Get a list of Docker images objects stored in the cache
        """
        return list(self.data.values())

    def getImageByName(self, name):
        """
        Get an image object using the Docker image name
        :param name: The docker image name
        :type name:string
        """
        imageKey = self._getHashKey(name)
        if imageKey in self.data:
            return self.data[imageKey]

    def isImageAlreadyLoaded(self, name):
        """

        Checks whether an image was already loaded, via docker image name.
        This check does not use the docker image id, and therefore will not
        treat two equivalent images with different names as similar
        :param name: The docker image name
        :type name:string
        """
        imageKey = self._getHashKey(name)
        return imageKey in self.data

    def _getHashKey(self, name):
        return DockerImage.getHashKey(name)

    def getRawData(self):
        dataCopy = {}
        for (key, val) in iteritems(self.data):
            dataCopy[key] = val.getRawData()
        return dataCopy

    def deleteImage(self, name):
        imageKey = self._getHashKey(name)
        if imageKey in self.data:
            del self.data[imageKey]
            return True
        else:
            return False

    def getAllCliSpec(self):
        spec_dict = {}

        for (key, val) in iteritems(self.data):
            spec_dict[val.name] = val.getCLIListSpec()

        return spec_dict
# TODO add regex for tag and digest names
# TODO add regex for clis to enforce alpha-numeric name


class DockerImageStructure:
    cli_schema = {
        'type': 'object',
        "properties": {
            DockerImage.type: {'type': 'string'},
            DockerImage.xml: {'type': 'string'}
        },
        'required': [DockerImage.type, DockerImage.xml],
        'additionalProperties': False
    }

    cli_list_schema = {
        'type': 'object',

        "patternProperties": {
            "^[a-zA-Z0-9_-]+$": cli_schema
        },
        # an image should have at least one cli
        'minProperties': 1,
        'additionalProperties': False
    }

    ImageSchema = {

        '$schema': 'http://json-schema.org/schema#',
        'type': 'object',
        'properties': {
            DockerImage.imageName: {'type': 'string'},
            DockerImage.imageHash: {'type': 'string'},
            DockerImage.cli_dict: cli_list_schema

        },
        'required': [DockerImage.imageName, DockerImage.imageHash,
                     DockerImage.cli_dict],
        'additionalProperties': True

    }
