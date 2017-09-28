import json
import six

from girder.constants import AccessType
from girder.utility.model_importer import ModelImporter
from girder import logger


def _getImageIdFromReference(identifier, reference):
    if not reference.get('taskId') or not reference.get('jobId'):
        logger.error('Event info did not contain the task or job id.')
        return

    # load the task definition from the task item
    taskId = reference['taskId']
    jobId = reference['jobId']
    task = ModelImporter.model('item').load(taskId, force=True)
    spec = task.get('meta', {}).get('itemTaskSpec')
    if not spec:
        logger.error('Could not get the task definition from the job.')
        return

    inputs = spec.get('inputs', [])
    outputs = spec.get('outputs', [])

    # get the definition of the current output parameter
    for output in outputs:
        if output.get('id') == identifier:
            break
    else:
        logger.error('Could not find an output associated with %s' % identifier)
        return

    # get the reference id associated with the output
    reference = output.get('extra', {}).get('reference')
    if not reference:
        # if no reference is provided, we fall back to an image input
        logger.warning('Annotation output did not provide a reference image')

    # find the definition of the referenced image
    for input in inputs:
        if reference and input.get('id') == reference:
            break
        elif not reference and input.get('type') == 'image':
            break
    else:
        logger.error('Could not determine an image to post the annotation to.')

    # get the input bindings to resolve the input image id
    job = ModelImporter.model('job', 'jobs').load(jobId, force=True)
    if not job:
        logger.error('Could not load the source job')
        return
    return job.get('itemTaskBindings', {}).get('inputs', {}).get(reference, {}).get('id')


def process_annotations(event):
    """Add annotations to an image on a ``data.process`` event"""
    info = event.info
    identifier = None
    reference = info.get('reference', None)
    if reference is not None:
        try:
            reference = json.loads(reference)
            if (isinstance(reference, dict) and
                    isinstance(reference.get('id'), six.string_types)):
                identifier = reference['id']
        except (ValueError, TypeError):
            logger.warning('Failed to parse data.process reference: %r', reference)

    if identifier is not None and identifier.endswith('AnnotationFile'):
        imageId = _getImageIdFromReference(identifier, reference)
        userId = str(info.get('currentUser', {}).get('_id'))
        if not imageId or not userId:
            logger.error('Annotation reference does not contain required information.')
            return

        # load model classes
        Item = ModelImporter.model('item')
        File = ModelImporter.model('file')
        User = ModelImporter.model('user')
        Annotation = ModelImporter.model('annotation', plugin='large_image')

        # load models from the database
        user = User.load(userId, force=True)
        item = Item.load(imageId, level=AccessType.READ, user=user)
        file = File.load(
            info.get('file', {}).get('_id'),
            level=AccessType.READ, user=user
        )

        if not (item and user and file):
            logger.error('Could not load models from the database')
            return

        try:
            data = json.loads(''.join(File.download(file)()))
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
