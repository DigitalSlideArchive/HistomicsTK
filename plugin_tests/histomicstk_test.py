#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import six
import time

from girder import config
from girder.constants import AccessType
from girder.models.group import Group
from girder.models.model_base import ValidationException
from girder.models.setting import Setting
from girder.models.user import User
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
    def setUp(self):
        base.TestCase.setUp(self)
        self.admin = User().findOne({'login': 'adminlogin'})
        self.user = User().findOne({'login': 'goodlogin'})
        self.user2 = User().findOne({'login': 'user2'})
        self.group = Group().createGroup('test group', creator=self.user2)
        Group().addUser(self.group, self.user2)

    def testHistomicsTKSettings(self):
        from girder.plugins.HistomicsTK.constants import PluginSettings

        key = PluginSettings.HISTOMICSTK_DEFAULT_DRAW_STYLES

        resp = self.request(path='/HistomicsTK/settings')
        self.assertStatusOk(resp)
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
        self.assertStatusOk(resp)
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
        }, {
            'key': PluginSettings.HISTOMICSTK_ANALYSIS_ACCESS,
            'initial': {'public': True},
            'bad': {
                '': 'must be a JSON object',
            },
            'badjson': [{
                'value': ['not an object'],
                'return': 'must be a JSON object',
            }, {
                'value': {
                    'public': False,
                    'users': [self.user, self.admin['_id']],
                    'groups': [self.user['_id']]},
                'return': 'No such group',
            }],
            'goodjson': [{
                'value': {
                    'public': False, 'users': [self.user, self.admin['_id']]},
                'return': {
                    'public': False, 'users': [self.user['_id'], self.admin['_id']]},
            }, {
                'value': {
                    'public': False, 'groups': [self.group]},
                'return': {
                    'public': False, 'groups': [self.group['_id']]},
            }],
        }]
        for setting in settings:
            key = setting['key']
            self.assertEqual(Setting().get(key), setting['initial'])
            for badval in setting.get('bad', {}):
                with six.assertRaisesRegex(self, ValidationException, setting['bad'][badval]):
                    Setting().set(key, badval)
            for badval in setting.get('badjosn', []):
                with six.assertRaisesRegex(self, ValidationException, badval['return']):
                    Setting().set(key, badval['value'])
            for goodval in setting.get('good', {}):
                self.assertEqual(Setting().set(key, goodval)['value'], setting['good'][goodval])
            for goodval in setting.get('goodjson', []):
                self.assertEqual(Setting().set(key, goodval['value'])['value'], goodval['return'])

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

    def testGetAnalysisAccess(self):
        from girder.plugins.HistomicsTK.constants import PluginSettings

        resp = self.request(path='/HistomicsTK/analysis/access', user=self.admin)
        self.assertStatusOk(resp)
        self.assertEqual(resp.json, {'public': True, 'users': [], 'groups': []})
        Setting().set(PluginSettings.HISTOMICSTK_ANALYSIS_ACCESS, {
            'public': False,
            'users': [self.user['_id'], self.admin['_id']],
            'groups': [self.group]})
        resp = self.request(path='/HistomicsTK/analysis/access', user=self.admin)
        self.assertStatusOk(resp)
        self.assertEqual(resp.json, {
            'public': False,
            'users': [{
                'id': str(self.user['_id']),
                'login': 'goodlogin',
                'name': 'First Last',
                'level': AccessType.READ,
            }, {
                'id': str(self.admin['_id']),
                'login': 'adminlogin',
                'name': 'Admin Last',
                'level': AccessType.READ,
            }],
            'groups': [{
                'id': str(self.group['_id']),
                'name': 'test group',
                'description': '',
                'level': AccessType.READ,
            }]})
        User().remove(self.user)
        Group().remove(self.group)
        resp = self.request(path='/HistomicsTK/analysis/access', user=self.admin)
        self.assertStatusOk(resp)
        self.assertEqual(resp.json, {
            'public': False,
            'users': [{
                'id': str(self.admin['_id']),
                'login': 'adminlogin',
                'name': 'Admin Last',
                'level': AccessType.READ,
            }],
            'groups': []})

    def testGetDockerImage(self):
        from girder.plugins.HistomicsTK.constants import PluginSettings

        # Add a CLI
        resp = self.request(
            path='/HistomicsTK/HistomicsTK/docker_image',
            user=self.admin, method='PUT',
            params={'name': '"girder/slicer_cli_web:small"'})
        self.assertStatusOk(resp)
        endTime = time.time() + 180  # maxTimeout
        while time.time() < endTime:
            try:
                resp = self.request(path='/HistomicsTK/HistomicsTK/docker_image', user=self.admin)
                if resp.output_status.startswith(b'200') and len(resp.json) == 1:
                    break
            except (AssertionError, KeyError):
                pass
            time.sleep(1)

        # Intially, access is public, so all users should see the entry,
        # but a missing user won't.
        # Setting().set(PluginSettings.HISTOMICSTK_ANALYSIS_ACCESS, {'public': True})
        for user, count in [(self.admin, 1), (self.user, 1), (self.user2, 1), (None, 0)]:
            resp = self.request(path='/HistomicsTK/HistomicsTK/docker_image', user=user)
            self.assertStatusOk(resp)
            self.assertEqual(len(resp.json), count)
        # Each user is allowed or in an allowed group
        Setting().set(PluginSettings.HISTOMICSTK_ANALYSIS_ACCESS, {
            'public': False,
            'users': [self.user['_id'], self.admin['_id']],
            'groups': [self.group]})
        for user, count in [(self.admin, 1), (self.user, 1), (self.user2, 1), (None, 0)]:
            resp = self.request(path='/HistomicsTK/HistomicsTK/docker_image', user=user)
            self.assertStatusOk(resp)
            self.assertEqual(len(resp.json), count)
        # Remove the first user's permissions
        Setting().set(PluginSettings.HISTOMICSTK_ANALYSIS_ACCESS, {
            'public': False, 'users': [], 'groups': [self.group]})
        for user, count in [(self.admin, 1), (self.user, 0), (self.user2, 1), (None, 0)]:
            resp = self.request(path='/HistomicsTK/HistomicsTK/docker_image', user=user)
            self.assertStatusOk(resp)
            self.assertEqual(len(resp.json), count)
        # Remove the group permissions
        Setting().set(PluginSettings.HISTOMICSTK_ANALYSIS_ACCESS, {
            'public': False, 'users': [], 'groups': []})
        for user, count in [(self.admin, 1), (self.user, 0), (self.user2, 0), (None, 0)]:
            resp = self.request(path='/HistomicsTK/HistomicsTK/docker_image', user=user)
            self.assertStatusOk(resp)
            self.assertEqual(len(resp.json), count)
