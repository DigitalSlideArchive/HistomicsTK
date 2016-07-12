import json

from girder.constants import TerminalColor, AccessType
from girder.utility.model_importer import ModelImporter


def process_annotations(event):
    """Add annotations to an image on a ``data.process`` event"""
    info = event.info
    if 'anot' in info.get('file', {}).get('exts', []):
        reference = info.get('reference', None)

        try:
            reference = json.loads(reference)
        except (ValueError, TypeError):
            print(TerminalColor.error(
                'Warning: Could not get reference from the annotation param. '
                'Make sure you have at ctk-cli>=1.3.1 installed.'
            ))
            return

        if 'userId' not in reference or 'itemId' not in reference:
            print(TerminalColor.error(
                'Annotation reference does not contain required information.'
            ))
            return

        userId = reference['userId']
        itemId = reference['itemId']

        # load model classes
        Item = ModelImporter.model('item')
        File = ModelImporter.model('file')
        User = ModelImporter.model('user')
        Annotation = ModelImporter.model('annotation', plugin='large_image')

        # load models from the database
        user = User.load(userId, force=True)
        item = Item.load(itemId, level=AccessType.WRITE, user=user)
        file = File.load(
            info.get('file', {}).get('_id'),
            level=AccessType.READ, user=user
        )

        if not (item and user and file):
            print(TerminalColor.error(
                'Could not load models from the database'
            ))
            return

        try:
            data = json.loads(
                ''.join(File.download(file)())
            )
        except Exception:
            print(TerminalColor.error(
                'Could not parse annotation file'
            ))
            return

        Annotation.createAnnotation(
            item,
            user,
            data
        )
