#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import six

from girder import config
from girder.models.model_base import ValidationException
from girder.models.setting import Setting
from tests import base


os.environ['GIRDER_PORT'] = os.environ.get('GIRDER_TEST_PORT', '20200')
config.loadConfig()  # Must reload config to pickup correct port


# boiler plate to start and stop the server
def setUpModule():
    base.enabledPlugins.append('HistomicsTK')
    base.startServer()


def tearDownModule():
    base.stopServer()


class HistomicsTKCoreTest(base.TestCase):

    def testSettings(self):
        from girder.plugins.HistomicsTK import constants

        key = constants.PluginSettings.HISTOMICSTK_DEFAULT_DRAW_STYLES

        resp = self.request(path='/HistomicsTK/settings')
        self.assertStatus(resp, 200)
        settings = resp.json
        self.assertEqual(settings[key], None)

        Setting().set(key, '')
        self.assertEqual(Setting().get(key), None)
        with six.assertRaisesRegex(self, ValidationException, 'must be a JSON'):
            Setting().set(key, 'not valid')
        with six.assertRaisesRegex(self, ValidationException, 'must be a JSON'):
            Setting().set(key, json.dumps({'not': 'a list'}))
        value = [{'lineWidth': 8, 'id': 'Group 8'}]
        Setting().set(key, json.dumps(value))
        self.assertEqual(json.loads(Setting().get(key)), value)
        Setting().set(key, value)
        self.assertEqual(json.loads(Setting().get(key)), value)

        resp = self.request(path='/HistomicsTK/settings')
        self.assertStatus(resp, 200)
        settings = resp.json
        self.assertEqual(json.loads(settings[key]), value)
