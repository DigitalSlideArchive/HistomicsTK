
from docker import Client
from docker.errors import DockerException
from girder.constants import AccessType
from girder.api.rest import getCurrentUser
from girder.models.model_base import ModelImporter, AccessControlledModel

import jsonschema

from ..models import DockerImage, DockerImageError, \
    DockerImageNotFoundError, DockerCache, DockerImageStructure

# from six import iteritems
# import os
# from lxml import etree
# from StringIO import StringIO


class Dockerimagemodel(AccessControlledModel):
    """
    Singleton class to manage access to cached docker image data. Data is
    retrieved as instances of either DockerImage or DockerCache objects
    """
    # TODO reference by image id or require image:digest
    imageHash = DockerImage.imageHash

    def initialize(self):
        self.name = 'dockerimagemodel'
        # use the DockerImage.gethash as the id
        self.ensureIndices([self.imageHash])
        self.exposeFields(AccessType.ADMIN, (DockerImage.imageHash,))
        self.versionId = None
        try:
            self.client = Client(base_url='unix://var/run/docker.sock')
        except DockerException as err:

            raise DockerImageError('could not create the docker '
                                   'client '+err.__str__())

    # TODO image_name:tag and image_name@digest are treated seperate images

    def putDockerImage(self, names, jobType, pullIfNotLocal=False):
        """
        Attempts to cache metadata on the docker images listed in the names
        list.
        If the pullIfNotLocal flag is true, the job will attempt to pull
         the image if it does not exist.
        :param names: A list of docker image names(can use with tags or digests)
        :param jobType: defines the jobtype of the job that will be schedueled
         ,used by event listeners to determine if a job succeeded or not
         :param pullIfNotLocal: Boolean to determine whether a non existent
         image
         should be pulled,(attempts to pull from default docker hub registry)
        """
        jobModel = ModelImporter.model('job', 'jobs')
        # list of images to pull and load
        pullList = []
        # list of images that exist locally and just need to be parsed and saved
        loadList = []
        for name in names:

            try:

                self._ImageExistsLocally(name)

                data = self.collection.find_one(DockerImage.getHashKey(name))

                if data is None:
                    loadList.append(name)
            # exception can be dockerimage
            except DockerImageNotFoundError:
                if pullIfNotLocal:
                    pullList.append(name)

        job = jobModel.createLocalJob(
            module='girder.plugins.HistomicsTK.image_job',
            function='jobPullAndLoad',
            kwargs={


                'pullList': pullList,
                'loadList': loadList
            },

            title='Pulling and caching docker images ',
            type=jobType,
            user=getCurrentUser(),
            public=True,
            async=True
        )

        jobModel.scheduleJob(job)

    def _ImageExistsLocally(self, name):
        """
        Checks if the docker image exist locally
        :param name: The name of the docker image

        :returns id: returns the docker image id
        """
        try:
            data = self.client.inspect_image(name)
        except Exception as err:

            raise DockerImageNotFoundError('could not find'
                                           ' the image \n'+err.__str__(), name)
        return data['Id']

    def save(self, img):
        """
        Attempt to save the docker image data in the mongo database
        :param img: An instance of a dockerImage object
        :type img: DockerImage

        """
        try:
            # rely on the parent model class to add a '_id' field
            super(Dockerimagemodel, self).save(document=img.getRawData(),
                                               triggerEvents=True)
        except Exception as err:
            raise DockerImageError(
                'Could not save image %s metadata '
                'to database ' % img.name + err.__str__(), img.name)

    def _load(self, imgHash):
        """
        Attempts to find a specific image in the girder mongo database
        :param imgHash: The hash of the image name used as a key the hash used
         is defined in the DockerImage class
         :type imgHash:string
        :returns:  The DockerImage instance was represented by the hash
        """
        results = super(Dockerimagemodel, self).findOne(
            {DockerImage.imageHash: imgHash})
        if results is None:
            raise DockerImageNotFoundError('The docker image with the hash'
                                           ' %s does not exist in the'
                                           ' database' % imgHash,
                                           None)
        return DockerImage(results)

    def _getAll(self):
        """
        Attempt to find all docker image data that was cached in the
        girder-mongo database
        :returns: a list of DockerImage objects
        """
        results = super(Dockerimagemodel, self).find()
        img_list = []

        for img in results:
            img_list.append(DockerImage(img))
        return img_list

    def saveAllImgs(self, dockerCache):
        """
        Attempts to same all images in the dockerCache ot the
        girder-mongo database
        :param dockerCache: A DockerCache object containing instances of
        DockerImages that are to be cached
        :type dockerCache:DockerCache
        """
        img_list = dockerCache.getImages()
        for val in img_list:
            self.save(val)

    def loadAllImages(self):
        """
        Attempts to generate a DockerCache object with all image meta data
        stored in girder.If DockerImage meta data is saved in girder but the
        actual docker image was deleted off the local machine, the meta data
        will be removed from the mongo database
        :returns: A DockerCache object populated with DockerImage objects
        """
        nonExist = []
        img_list = self._getAll()
        dockerCache = DockerCache()
        for img in img_list:
            try:
                self._ImageExistsLocally(img.name)
                dockerCache.addImage(img)
            except DockerImageNotFoundError:
                nonExist.append(img.name)
        self.removeImages(nonExist)
        return dockerCache

    def delete_docker_image_from_repo(self, name, jobType):
        """
        Creates an asynchronous job to delete the docker images listed in name
        from the local machine
        :param name:A list of docker image names
        :type name: list of strings
        :param jobType: the value to use for the job's type. This is used by
        event listeners to determine which jobs are related to the DockerImages
        """

        jobModel = ModelImporter.model('job', 'jobs')

        job = jobModel.createLocalJob(
            module='girder.plugins.HistomicsTK.image_job',
            function='deleteImage',
            kwargs={
                'deleteList': name
            },
            title='Deleting Docker Images',
            user=getCurrentUser(),
            type=jobType,
            public=True,
            async=True
        )

        jobModel.scheduleJob(job)

    def removeImages(self, imgList):
        """
        Attempt to remove image metadata from the mongo database
        :param imgList: a list of docker image names
        :type imgList: a list of strings

        """
        try:
            for img in imgList:
                hash = DockerImage.getHashKey(img)
                imageData = self._load(hash)
                super(Dockerimagemodel, self).remove(imageData.getRawData())
        except Exception as err:
            if isinstance(err, DockerImageNotFoundError):
                raise DockerImageNotFoundError(
                    'The image %s with hash %s does not exist '
                    'in the database' % (img, hash), img)
            else:
                raise DockerImageError('Could not delete the image '
                                       'data from the database invalid '
                                       'image :'+img+' ' +
                                       err.__str__(), img)

    # TODO validate the xml of each cli
    def validate(self, doc):
        try:
            # validate structure of cached data on docker image
            jsonschema.validate(doc, DockerImageStructure.ImageSchema)
            # check cli xml is correct
            #
            # loc=os.path.dirname(os.path.abspath(__file__))+'/ModuleDescription.xsd'
            # schemaFile = open(loc)
            #
            # schemaData = schemaFile.read()
            #
            # schema_doc = etree.parse(StringIO(schemaData))
            # schema = etree.XMLSchema(schema_doc)
            #
            # for (key, val) in iteritems(doc[DockerImage.cli_dict]):
            #     xml = val[DockerImage.xml]
            #     cli_xml = etree.parse(xml)
            #     schema.assertValid(cli_xml)
            #
            return doc
        except Exception as err:

            raise DockerImageError('Image meta data is invalid ' + err.message)
