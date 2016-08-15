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

# This is to serve as an example for how to create a server-side test in a
# girder plugin, it is not meant to be useful.

from tests import base

from girder import events
import threading

import types
import json
# boiler plate to start and stop the server
TIMEOUT = 300


def setUpModule():
    base.enabledPlugins.append('HistomicsTK')
    base.startServer()
    global JobStatus
    from girder.plugins.jobs.constants import JobStatus


def tearDownModule():
    base.stopServer()

# TODO when endpoint information is added to the get endpoint
# TODO add tests for querying cli xml and running clis


class HistomicsTKExampleTest(base.TestCase):
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

    def testBadImageAdd(self):
        # add a bad image
        img_name = 'null/null:null'
        self.noImages()
        self.addImage(img_name, JobStatus.ERROR)
        self.noImages()

    def testDockerAdd(self):
        # try to cache a good image to the mongo database
        img_name = "dsarchive/histomicstk:v0.1.3"
        self.noImages()
        self.addImage(img_name, JobStatus.SUCCESS)
        self.imageIsLoaded(img_name, True)

    def testDockerDelete(self):
        # just delete the meta data in the mongo database
        # dont attempt to delete the docker image
        img_name = "dsarchive/histomicstk:v0.1.3"
        self.noImages()
        self.addImage(img_name, JobStatus.SUCCESS)
        self.imageIsLoaded(img_name, True)
        self.deleteImage(img_name, True, False)
        self.imageIsLoaded(img_name, exists=False)
        self.noImages()

    def testDockerDeleteFull(self):
        # attempt to delete docker image metadata and the image off the local
        # machine
        img_name = "dsarchive/histomicstk:v0.1.3"
        self.noImages()
        self.addImage(img_name, JobStatus.SUCCESS)
        self.imageIsLoaded(img_name, True)
        self.deleteImage(img_name, True, True, JobStatus.SUCCESS)
        self.imageIsLoaded(img_name, exists=False)
        self.noImages()

    def testDockerPull(self):

        # test an instance when the image must be pulled
        # Forces the test image to be deleted
        self.testDockerDeleteFull()
        self.testDockerAdd()

    def testBadImageDelete(self):
        # attempt to delete a non existent image
        img_name = 'null/null:null'
        self.noImages()
        self.deleteImage(img_name, False, )

    def imageIsLoaded(self, name, exists):
        resp = self.request(path='/HistomicsTK/HistomicsTK/docker_image',
                            user=self.admin)

        data = json.loads(self.getBody(resp))
        if not exists:
            self.assertNotHasKeys(data, [name])

        else:
            self.assertHasKeys(data, [name])

    def noImages(self):
        resp = self.request(path='/HistomicsTK/HistomicsTK/docker_image',
                            user=self.admin)
        self.assertStatus(resp, 200)
        self.assertEqual('{}', self.getBody(resp),
                         " There should be no pre existing docker images ")

    def deleteImage(self, name,  responseCodeOK, deleteDockerImage=False,
                    status=4):
        """
        delete docker image data and test whether a docker
        image can be deleted off the local machine
        """
        if deleteDockerImage:
            event = threading.Event()

            def tempListener(self, girderEvent):
                job = girderEvent.info

                if job['type'] == 'HistomicsTK_job' and \
                        (job['status'] == JobStatus.SUCCESS or
                         job['status'] == JobStatus.ERROR):

                    self.assertEqual(job['status'], status,
                                     "The status of the job should "
                                     "match")
                    events.unbind('model.job.save.after', 'HistomicsTK_del')
                    # del self.delHandler
                    event.set()

            self.delHandler = types.MethodType(tempListener, self)

            events.bind('model.job.save.after', 'HistomicsTK_del',
                        self.delHandler)

        resp = self.request(path='/HistomicsTK/HistomicsTK/docker_image',
                            user=self.admin, method='DELETE',
                            params={"name": json.dumps(name),
                                    "delete_from_local_repo":
                                        deleteDockerImage
                                    }, isJson=False)
        if responseCodeOK:
            self.assertStatus(resp, 200)
        else:
            try:
                self.assertStatus(resp, 200)
                self.fail('A status ok or code 200 should not have been '
                          'recieved for deleting the image %s' % str(name))
            except Exception:
                    pass
        if deleteDockerImage:
            if not event.wait(TIMEOUT):
                self.fail('deleting the docker image is taking '
                          'longer than %d seconds' % TIMEOUT)

            del self.delHandler

    def addImage(self, name, status):
        """test the put endpoint, name can be a string or a list of strings"""

        event = threading.Event()

        def tempListener(self, girderEvent):

            job = girderEvent.info

            if job['type'] == 'HistomicsTK_job' and \
                    (job['status'] == JobStatus.SUCCESS or
                     job['status'] == JobStatus.ERROR):

                self.assertEqual(job['status'], status,
                                 "The status of the job should "
                                 "match ")

                events.unbind('model.job.save.after', 'HistomicsTK_add')

                event.set()

        self.addHandler = types.MethodType(tempListener, self)

        events.bind('model.job.save.after', 'HistomicsTK_add', self.addHandler)

        resp = self.request(path='/HistomicsTK/HistomicsTK/docker_image',
                            user=self.admin, method='PUT',
                            params={"name": json.dumps(name)}, isJson=False)

        self.assertStatus(resp, 200)

        if not event.wait(TIMEOUT):
            self.fail('adding the docker image is taking '
                      'longer than %d seconds' % TIMEOUT)
        del self.addHandler
