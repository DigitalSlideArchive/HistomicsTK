from girder.api import access
from girder.api.v1.item import Item as ItemResource
from girder.api.describe import autoDescribeRoute, Description
from girder.constants import AccessType
from girder.models.folder import Folder


class ImageBrowseResource(ItemResource):
    """Extends the "item" resource to iterate through images im a folder."""

    def __init__(self, apiRoot):
        # Don't call the parent (Item) constructor, to avoid redefining routes,
        # but do call the grandparent (Resource) constructor
        super(ItemResource, self).__init__()

        self.resourceName = 'item'
        apiRoot.item.route('GET', (':id', 'next_image'), self.getNextImage)
        apiRoot.item.route('GET', (':id', 'previous_image'), self.getPreviousImage)

    def getAdjacentImages(self, currentImage):
        folderModel = Folder()
        folder = folderModel.load(
            currentImage['folderId'], user=self.getCurrentUser(), level=AccessType.READ)
        allImages = list(folderModel.childItems(folder))
        index = allImages.index(currentImage)
        return {
            'previous': allImages[index - 1],
            'next': allImages[(index + 1) % len(allImages)]
        }

    @access.public
    @autoDescribeRoute(
        Description('Get the next image in the same folder as the given item.')
        .modelParam('id', 'The current image ID',
                    model='item', destName='image', paramType='path', level=AccessType.READ)
        .errorResponse()
        .errorResponse('Image not found', code=404)
    )
    def getNextImage(self, image):
        return self.getAdjacentImages(image)['next']

    @access.public
    @autoDescribeRoute(
        Description('Get the previous image in the same folder as the given item.')
        .modelParam('id', 'The current item ID',
                    model='item', destName='image', paramType='path', level=AccessType.READ)
        .errorResponse()
        .errorResponse('Image not found', code=404)
    )
    def getPreviousImage(self, image):
        return self.getAdjacentImages(image)['previous']
