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

import json

from girder import events

from tests import base


# boiler plate to start and stop the server
def setUpModule():
    base.enabledPlugins.append('HistomicsTK')
    base.startServer()


def tearDownModule():
    base.stopServer()


class AnnotationHandlerTest(base.TestCase):

    def testHandleAnnotation(self):
        admin = self.model('user').findOne({'login': 'admin'})
        item = self.model('item').findOne({'name': 'Item 1'})
        file1 = self.model('file').findOne({'name': 'File 1'})
        file2 = self.model('file').findOne({'name': 'File 2'})
        file3 = self.model('file').findOne({'name': 'File 3'})
        file4 = self.model('file').findOne({'name': 'File 4'})
        assetstore = self.model('assetstore').load(id=file1['assetstoreId'])
        annot = list(self.model('annotation', 'large_image').find({'itemId': item['_id']}))
        self.assertEqual(len(annot), 0)

        # Process a list of annotations
        events.trigger('data.process', {
            'file': file1,
            'assetstore': assetstore,
            'reference': json.dumps({
                'identifier': 'sampleAnnotationFile',
                'itemId': str(file1['_id']),
                'userId': str(admin['_id']),
            })
        })
        annot = list(self.model('annotation', 'large_image').find({'itemId': item['_id']}))
        self.assertEqual(len(annot), 2)

        # If the reference doesn't contain userId or itemId, we won't add any
        # annotations
        events.trigger('data.process', {
            'file': file1,
            'assetstore': assetstore,
            'reference': json.dumps({
                'identifier': 'sampleAnnotationFile',
                'userId': str(admin['_id']),
            })
        })
        annot = list(self.model('annotation', 'large_image').find({'itemId': item['_id']}))
        self.assertEqual(len(annot), 2)
        events.trigger('data.process', {
            'file': file1,
            'assetstore': assetstore,
            'reference': json.dumps({
                'identifier': 'sampleAnnotationFile',
                'itemId': str(file1['_id']),
            })
        })
        annot = list(self.model('annotation', 'large_image').find({'itemId': item['_id']}))
        self.assertEqual(len(annot), 2)

        # If the user id isn't valid, we won't add an annotation
        events.trigger('data.process', {
            'file': file1,
            'assetstore': assetstore,
            'reference': json.dumps({
                'identifier': 'sampleAnnotationFile',
                'itemId': str(file1['_id']),
                'userId': str(item['_id']),  # this is not a user
            })
        })
        annot = list(self.model('annotation', 'large_image').find({'itemId': item['_id']}))
        self.assertEqual(len(annot), 2)

        # Process a single annotation
        events.trigger('data.process', {
            'file': file2,
            'assetstore': assetstore,
            'reference': json.dumps({
                'identifier': 'sampleAnnotationFile',
                'itemId': str(file2['_id']),
                'userId': str(admin['_id']),
            })
        })
        annot = list(self.model('annotation', 'large_image').find({'itemId': item['_id']}))
        self.assertEqual(len(annot), 3)

        # A file that isn't json shouldn't throw an error or add anything
        with self.assertRaises(ValueError):
            events.trigger('data.process', {
                'file': file3,
                'assetstore': assetstore,
                'reference': json.dumps({
                    'identifier': 'sampleAnnotationFile',
                    'itemId': str(file3['_id']),
                    'userId': str(admin['_id']),
                })
            })
        annot = list(self.model('annotation', 'large_image').find({'itemId': item['_id']}))
        self.assertEqual(len(annot), 3)

        # A json file that isn't an annotation shouldn't add anything either
        with self.assertRaises(AttributeError):
            events.trigger('data.process', {
                'file': file4,
                'assetstore': assetstore,
                'reference': json.dumps({
                    'identifier': 'sampleAnnotationFile',
                    'itemId': str(file4['_id']),
                    'userId': str(admin['_id']),
                })
            })
        annot = list(self.model('annotation', 'large_image').find({'itemId': item['_id']}))
        self.assertEqual(len(annot), 3)
