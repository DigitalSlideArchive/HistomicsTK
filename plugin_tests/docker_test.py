#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
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
###############################################################################

from tests import base


def setUpModule():
    base.enabledPlugins.append('HistomicsTK')
    base.startServer()


def tearDownModule():
    base.stopServer()


class DockerImageEndpointTest(base.TestCase):
    # This only tests for the existance of the expected endpoints.  The code
    # behind the docker_images is tested by the slicer_cli_web plugin
    def setUp(self):
        # adding and removing docker images and using generated rest endpoints
        # requires admin access
        base.TestCase.setUp(self)
        admin = {
            'email': 'admin@email.com',
            'login': 'adminlogin',
            'firstName': 'Admin',
            'lastName': 'Last',
            'password': 'adminpassword',
            'admin': True
        }
        self.admin = self.model('user').createUser(**admin)

    def testGetEndpoint(self):
        resp = self.request(path='/HistomicsTK/HistomicsTK/docker_image',
                            user=self.admin)
        self.assertStatusOk(resp)
        # We should have no images
        self.assertEqual(resp.json, {})

    def testPutEndpoint(self):
        resp = self.request(path='/HistomicsTK/HistomicsTK/docker_image',
                            user=self.admin, method='PUT')
        # We should get back an error code since we didn't send some parameters
        self.assertStatus(resp, 400)

    def testDeleteEndpoint(self):
        resp = self.request(path='/HistomicsTK/HistomicsTK/docker_image',
                            user=self.admin, method='DELETE')
        # We should get back an error code since we didn't send some parameters
        self.assertStatus(resp, 400)
