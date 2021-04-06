import cherrypy
import json
import os

from girder.models.assetstore import Assetstore
from girder.models.folder import Folder
from girder.models.item import Item
from girder.models.user import User
from girder.utility.progress import ProgressContext
from girder.utility.server import configureServer

from girder_large_image_annotation.models.annotation import Annotation

from datastore import datastore


def namedFolder(user, folderName='Public'):
    return Folder().find({
        'parentId': user['_id'],
        'name': folderName,
    })[0]


cherrypy.config['database']['uri'] = 'mongodb://mongodb:27017/girder'

# This loads plugins, allowing setting validation
configureServer()

if User().findOne() is None:
    User().createUser('admin', 'password', 'Admin', 'Admin', 'admin@nowhere.nil')
adminUser = User().findOne({'admin': True})
if Assetstore().findOne() is None:
    Assetstore().createFilesystemAssetstore('Assetstore', '/assetstore')
fsAssetstore = Assetstore().findOne()
# Upload a list of items
publicFolder = namedFolder(adminUser)
dataFiles = {
    'tcga1': {
        'name': 'TCGA-A2-A0YE-01Z-00-DX1.8A2E3094-5755-42BC-969D-7F0A2ECA0F39.svs'},
}
for key, entry in dataFiles.items():
    path = datastore.fetch(entry['name'])
    query = {'folderId': publicFolder['_id'], 'name': os.path.basename(path)}
    if not Item().findOne(query):
        with ProgressContext(False, user=adminUser) as ctx:
            Assetstore().importData(
                fsAssetstore, publicFolder, 'folder', {'importPath': path},
                ctx, adminUser, leafFoldersAsItems=False)
    entry['item'] = Item().findOne(query)
    print(entry['item'])
# Add annotations
annotationFiles = {
    'tcga1': {
        'item': 'tcga1',
        'name': 'TCGA-A2-A0YE-01Z-00-DX1.8A2E3094-5755-42BC-969D-'
                '7F0A2ECA0F39.svs_annotations.json',
    }
}
for key, entry in annotationFiles.items():
    item = dataFiles[entry['item']]['item']
    query = {'_active': {'$ne': False}, 'itemId': item['_id']}
    if not Annotation().findOne(query):
        path = datastore.fetch(entry['name'])
        annotations = json.load(open(path))
        if not isinstance(annotations, list):
            annotations = [annotations]
        for annotation in annotations:
            Annotation().createAnnotation(
                item, adminUser, annotation.get('annotation', annotation))
    print(Annotation().find(query).count())
