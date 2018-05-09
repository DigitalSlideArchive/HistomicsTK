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

    def testHistomicsTKSettings(self):
        from girder.plugins.HistomicsTK.constants import PluginSettings

        key = PluginSettings.HISTOMICSTK_DEFAULT_DRAW_STYLES

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

    def testGeneralSettings(self):
        from girder.plugins.HistomicsTK.constants import PluginSettings

        settings = [{
            'key': PluginSettings.HISTOMICSTK_WEBROOT_PATH,
            'initial': 'histomicstk',
            'bad': {
                'girder': 'not be "girder"',
                '': 'not be empty'
            },
            'good': {'alternate1': 'alternate1'},
        }, {
            'key': PluginSettings.HISTOMICSTK_BRAND_NAME,
            'initial': 'HistomicsTK',
            'bad': {'': 'not be empty'},
            'good': {'Alternate': 'Alternate'},
        }, {
            'key': PluginSettings.HISTOMICSTK_BRAND_COLOR,
            'initial': '#777777',
            'bad': {
                '': 'not be empty',
                'white': 'be a hex color',
                '#777': 'be a hex color'
            },
            'good': {'#000000': '#000000'},
        }, {
            'key': PluginSettings.HISTOMICSTK_BANNER_COLOR,
            'initial': '#f8f8f8',
            'bad': {
                '': 'not be empty',
                'white': 'be a hex color',
                '#777': 'be a hex color'
            },
            'good': {'#000000': '#000000'},
        }]
        for setting in settings:
            key = setting['key']
            self.assertEqual(Setting().get(key), setting['initial'])
            for badval in setting['bad']:
                with six.assertRaisesRegex(self, ValidationException, setting['bad'][badval]):
                    Setting().set(key, badval)
            for goodval in setting['good']:
                self.assertEqual(Setting().set(key, goodval)['value'], setting['good'][goodval])

    def testGetWebroot(self):
        from girder.plugins.HistomicsTK.constants import PluginSettings

        resp = self.request(path='/histomicstk', method='GET', isJson=False, prefix='')
        self.assertStatusOk(resp)
        body = self.getBody(resp)
        assert '<title>HistomicsTK</title>' in body
        resp = self.request(path='/alternate2', method='GET', isJson=False, prefix='')
        self.assertStatus(resp, 404)
        Setting().set(PluginSettings.HISTOMICSTK_WEBROOT_PATH, 'alternate2')
        Setting().set(PluginSettings.HISTOMICSTK_BRAND_NAME, 'Alternate')
        resp = self.request(path='/histomicstk', method='GET', isJson=False, prefix='')
        self.assertStatusOk(resp)
        body = self.getBody(resp)
        assert '<title>Alternate</title>' in body
        resp = self.request(path='/alternate2', method='GET', isJson=False, prefix='')
        self.assertStatusOk(resp)
        body = self.getBody(resp)
        assert '<title>Alternate</title>' in body
