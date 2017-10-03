#!/usr/bin/env python
# -*- coding: utf-8 -*-

#############################################################################
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
#############################################################################

# This is to serve as an example for how to create a server-side test in a
# girder plugin, it is not meant to be useful.

import os
import json

from mock import patch
from girder import events

from tests import base


# boiler plate to start and stop the server
def setUpModule():
    base.enabledPlugins.append('HistomicsTK')
    base.startServer()


def tearDownModule():
    base.stopServer()


class AnnotationHandlerTest(base.TestCase):

    def generateItemTaskFromJson(self, fileName):
        filePath = os.path.join(
            os.path.dirname(__file__),
            fileName
        )
        with open(filePath) as f:
            spec = f.read()
        admin = self.model('user').findOne({'login': 'admin'})
        folder = self.model('folder').findOne({'name': 'Tasks'})
        item = self.model('item').createItem(
            name=fileName,
            creator=admin,
            folder=folder
        )
        token = self.model('token').createToken(
            days=1, scope='item_task.set_task_spec.%s' % str(item['_id'])
        )
        resp = self.request(
            '/item/%s/item_task_json_specs' % item['_id'],
            method='PUT', params={
                'json': spec,
                'image': fileName,
                'taskName': "NucleiDetection",
                'pullImage': False
            }, token=token
        )
        self.assertStatusOk(resp)
        return item

    def generateMockedJob(self, task, inputs, outputs):
        admin = self.model('user').findOne({'login': 'admin'})
        with patch('girder.plugins.jobs.models.job.Job.scheduleJob'):
            resp = self.request(
                '/item_task/%s/execution' % task['_id'],
                method='POST',
                params={
                    'inputs': json.dumps(inputs),
                    'outputs': json.dumps(outputs)
                },
                user=admin
            )
        self.assertStatusOk(resp)
        return resp.json

    def testHandleAnntation(self):
        admin = self.model('user').findOne({'login': 'admin'})
        item = self.model('item').findOne({'name': 'Item 1'})
        file1 = self.model('file').findOne({'name': 'File 1'})
        file2 = self.model('file').findOne({'name': 'File 2'})
        file3 = self.model('file').findOne({'name': 'File 3'})
        file4 = self.model('file').findOne({'name': 'File 4'})
        assetstore = self.model('assetstore').load(id=file1['assetstoreId'])
        annot = list(self.model('annotation', 'large_image').find({'itemId': item['_id']}))
        self.assertEqual(len(annot), 0)
        task = self.generateItemTaskFromJson('annotation_task.json')
        token = self.model('token').createToken(admin)

        inputs = {
            'inputImageFile': {
                'mode': 'girder',
                'resource_type': 'item',
                'id': str(item['_id']),
                'fileName': None
            }
        }
        outputs = {
            'outputNucleiAnnotationFile': {
                'mode': 'girder',
                'parent_type': 'folder',
                'parent_id': str(item['folderId']),
                'name': 'annotations'
            }
        }
        job = self.generateMockedJob(task, inputs, outputs)

        # Process a list of annotations
        events.trigger('data.process', {
            'file': file1,
            'assetstore': assetstore,
            'reference': json.dumps({
                'type': 'item_tasks.output',
                'id': 'outputNucleiAnnotationFile',
                'taskId': str(task['_id']),
                'itemId': str(file1['_id']),
                'jobId': str(job['_id'])
            }),
            'currentToken': token,
            'currentUser': admin
        })
        annot = list(self.model('annotation', 'large_image').find({'itemId': item['_id']}))
        self.assertEqual(len(annot), 2)

        # If the reference doesn't contain a taskId, we won't add any
        # annotations
        events.trigger('data.process', {
            'file': file1,
            'assetstore': assetstore,
            'reference': json.dumps({
                'type': 'item_tasks.output',
                'id': 'outputNucleiAnnotationFile',
                'itemId': str(file1['_id']),
                'jobId': str(job['_id'])
            }),
            'currentToken': token,
            'currentUser': admin
        })
        annot = list(self.model('annotation', 'large_image').find({'itemId': item['_id']}))
        self.assertEqual(len(annot), 2)

        # Process a single annotation
        events.trigger('data.process', {
            'file': file2,
            'assetstore': assetstore,
            'reference': json.dumps({
                'type': 'item_tasks.output',
                'id': 'outputNucleiAnnotationFile',
                'taskId': str(task['_id']),
                'itemId': str(file2['_id']),
                'jobId': str(job['_id'])
            }),
            'currentToken': token,
            'currentUser': admin
        })
        annot = list(self.model('annotation', 'large_image').find({'itemId': item['_id']}))
        self.assertEqual(len(annot), 3)

        # A file that isn't json shouldn't throw an error or add anything
        with self.assertRaises(ValueError):
            events.trigger('data.process', {
                'file': file3,
                'assetstore': assetstore,
                'reference': json.dumps({
                    'type': 'item_tasks.output',
                    'id': 'outputNucleiAnnotationFile',
                    'taskId': str(task['_id']),
                    'itemId': str(file3['_id']),
                    'jobId': str(job['_id'])
                }),
                'currentToken': token,
                'currentUser': admin
            })
        annot = list(self.model('annotation', 'large_image').find({'itemId': item['_id']}))
        self.assertEqual(len(annot), 3)

        # A json file that isn't an annotation shouldn't add anything either
        with self.assertRaises(AttributeError):
            events.trigger('data.process', {
                'file': file4,
                'assetstore': assetstore,
                'reference': json.dumps({
                    'type': 'item_tasks.output',
                    'id': 'outputNucleiAnnotationFile',
                    'taskId': str(task['_id']),
                    'itemId': str(file4['_id']),
                    'jobId': str(job['_id'])
                }),
                'currentToken': token,
                'currentUser': admin
            })
        annot = list(self.model('annotation', 'large_image').find({'itemId': item['_id']}))
        self.assertEqual(len(annot), 3)
