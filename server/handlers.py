import json
import six

from girder.constants import AccessType
from girder.utility.model_importer import ModelImporter
from girder import logger


def process_annotations(event):
    """Add annotations to an image on a ``data.process`` event"""
    info = event.info
    identifier = None
    reference = info.get('reference', None)
    if reference is not None:
        try:
            reference = json.loads(reference)
            if (isinstance(reference, dict) and
                    isinstance(reference.get('identifier'), six.string_types)):
                identifier = reference['identifier']
        except (ValueError, TypeError):
            logger.warning('Failed to parse data.process reference: %r', reference)
    if identifier is not None and identifier.endswith('AnnotationFile'):
        if 'userId' not in reference or 'itemId' not in reference:
            logger.error('Annotation reference does not contain required information.')
            return

        userId = reference['userId']
        imageId = reference['itemId']

        # load model classes
        Item = ModelImporter.model('item')
        File = ModelImporter.model('file')
        User = ModelImporter.model('user')
        Annotation = ModelImporter.model('annotation', plugin='large_image')

        # load models from the database
        user = User.load(userId, force=True)
        image = File.load(imageId, level=AccessType.READ, user=user)
        item = Item.load(image['itemId'], level=AccessType.READ, user=user)
        file = File.load(
            info.get('file', {}).get('_id'),
            level=AccessType.READ, user=user
        )

        if not (item and user and file):
            logger.error('Could not load models from the database')
            return

        try:
            data = json.loads(b''.join(File.download(file)()).decode('utf8'))
        except Exception:
            logger.error('Could not parse annotation file')
            raise

        if not isinstance(data, list):
            data = [data]
        for annotation in data:
            try:
                Annotation.createAnnotation(item, user, annotation)
            except Exception:
                logger.error('Could not create annotation object from data')
                raise
