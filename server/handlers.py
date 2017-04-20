import json

from girder.constants import AccessType
from girder.utility.model_importer import ModelImporter
from girder import logger


def process_annotations(event):
    """Add annotations to an image on a ``data.process`` event"""
    info = event.info
    if 'anot' in info.get('file', {}).get('exts', []):
        reference = info.get('reference', None)

        try:
            reference = json.loads(reference)
        except (ValueError, TypeError):
            logger.error(
                'Warning: Could not get reference from the annotation param. '
                'Make sure you have at least ctk-cli>=1.4.1 installed.'
            )
            raise

        if 'userId' not in reference or 'itemId' not in reference:
            logger.error(
                'Annotation reference does not contain required information.'
            )
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
            logger.error(
                'Could not load models from the database'
            )
            return

        try:
            data = json.loads(
                ''.join(File.download(file)())
            )
        except Exception:
            logger.error(
                'Could not parse annotation file'
            )
            raise

        try:
            Annotation.createAnnotation(
                item,
                user,
                data
            )
        except Exception:
            logger.error(
                'Could not create annotation object from data'
            )
            raise
