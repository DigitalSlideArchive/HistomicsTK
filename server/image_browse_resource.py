from girder.api import access
from girder.api.v1.item import Item as ItemResource
from girder.api.describe import autoDescribeRoute, Description
from girder.constants import AccessType
from girder.exceptions import RestException
from girder.models.folder import Folder


def _isLargeImageItem(item):
    return item.get('largeImage', {}).get('fileId') is not None


class ImageBrowseResource(ItemResource):
    """Extends the "item" resource to iterate through images im a folder."""

    def __init__(self, apiRoot):
        # Don't call the parent (Item) constructor, to avoid redefining routes,
        # but do call the grandparent (Resource) constructor
        super(ItemResource, self).__init__()

        self.resourceName = 'item'
        apiRoot.item.route('GET', (':id', 'next_image'), self.getNextImage)
        apiRoot.item.route('GET', (':id', 'previous_image'), self.getPreviousImage)

    def getAdjacentImages(self, currentImage, currentFolder=None):
        folderModel = Folder()
        if currentFolder:
            folder = currentFolder
        else:
            folder = folderModel.load(
                currentImage['folderId'], user=self.getCurrentUser(), level=AccessType.READ)

        if folder.get('isVirtual'):
            children = folderModel.childItems(folder, includeVirtual=True)
        else:
            children = folderModel.childItems(folder)

        allImages = [item for item in children if _isLargeImageItem(item)]
        try:
            index = allImages.index(currentImage)
        except ValueError:
            raise RestException('Id is not an image', 404)

        return {
            'previous': allImages[index - 1],
            'next': allImages[(index + 1) % len(allImages)]
        }

    @access.public
    @autoDescribeRoute(
        Description('Get the next image in the same folder as the given item.')
        .modelParam('id', 'The current image ID',
                    model='item', destName='image', paramType='path', level=AccessType.READ)
        .modelParam('folderId', 'The (virtual) folder ID the image is located in',
                    model='folder', destName='folder', paramType='query', level=AccessType.READ,
                    required=False)
        .errorResponse()
        .errorResponse('Image not found', code=404)
    )
    def getNextImage(self, image, folder):
        return self.getAdjacentImages(image, folder)['next']

    @access.public
    @autoDescribeRoute(
        Description('Get the previous image in the same folder as the given item.')
        .modelParam('id', 'The current item ID',
                    model='item', destName='image', paramType='path', level=AccessType.READ)
        .modelParam('folderId', 'The (virtual) folder ID the image is located in',
                    model='folder', destName='folder', paramType='query', level=AccessType.READ,
                    required=False)
        .errorResponse()
        .errorResponse('Image not found', code=404)
    )
    def getPreviousImage(self, image, folder):
        return self.getAdjacentImages(image, folder)['previous']
