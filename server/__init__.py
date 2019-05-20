import datetime
import json
import os
import re

from girder import events
from girder.api import access
from girder.api.describe import Description, describeRoute, autoDescribeRoute
from girder.api.rest import filtermodel
from girder.constants import AccessType, SettingDefault, TokenScope
from girder.exceptions import RestException, ValidationException
from girder.models.folder import Folder
from girder.models.group import Group
from girder.models.item import Item
from girder.models.setting import Setting
from girder.models.user import User
from girder.utility.webroot import Webroot
from girder.utility import config, setting_utilities

from girder.plugins.slicer_cli_web.rest_slicer_cli import (
    genRESTEndPointsForSlicerCLIsInDockerCache
)

from girder.plugins.slicer_cli_web.docker_resource import DockerResource

from .handlers import process_annotations
from .constants import PluginSettings
from .image_browse_resource import ImageBrowseResource
from . import ctk_cli_adjustment  # noqa - for side effects

from girder.models.model_base import ModelImporter


_template = os.path.join(
    os.path.dirname(__file__),
    'webroot.mako'
)


@setting_utilities.validator({
    PluginSettings.HISTOMICSTK_DEFAULT_DRAW_STYLES
})
def validateListOrJSON(doc):
    val = doc['value']
    try:
        if isinstance(val, list):
            doc['value'] = json.dumps(val)
        elif val is None or val.strip() == '':
            doc['value'] = None
        else:
            parsed = json.loads(val)
            if not isinstance(parsed, list):
                raise ValueError
            doc['value'] = val.strip()
    except (ValueError, AttributeError):
        raise ValidationException('%s must be a JSON list.' % doc['key'], 'value')


@setting_utilities.validator({
    PluginSettings.HISTOMICSTK_BANNER_COLOR,
    PluginSettings.HISTOMICSTK_BRAND_COLOR,
})
def validateHistomicsTKColor(doc):
    if not doc['value']:
        raise ValidationException('The banner color may not be empty', 'value')
    elif not re.match(r'^#[0-9A-Fa-f]{6}$', doc['value']):
        raise ValidationException('The banner color must be a hex color triplet', 'value')


@setting_utilities.validator(PluginSettings.HISTOMICSTK_BRAND_NAME)
def validateHistomicsTKBrandName(doc):
    if not doc['value']:
        raise ValidationException('The brand name may not be empty', 'value')


@setting_utilities.validator(PluginSettings.HISTOMICSTK_WEBROOT_PATH)
def validateHistomicsTKWebrootPath(doc):
    if not doc['value']:
        raise ValidationException('The webroot path may not be empty', 'value')
    if re.match(r'^girder$', doc['value']):
        raise ValidationException('The webroot path may not be "girder"', 'value')


@setting_utilities.validator(PluginSettings.HISTOMICSTK_ANALYSIS_ACCESS)
def validateHistomicsTKAnalysisAccess(doc):
    value = doc['value']
    if not isinstance(value, dict):
        raise ValidationException('Analysis access policy must be a JSON object.')
    for i, groupId in enumerate(value.get('groups', ())):
        if isinstance(groupId, dict):
            groupId = groupId.get('_id', groupId.get('id'))
        group = Group().load(groupId, force=True, exc=True)
        value['groups'][i] = group['_id']
    for i, userId in enumerate(value.get('users', ())):
        if isinstance(userId, dict):
            userId = userId.get('_id', userId.get('id'))
        user = User().load(userId, force=True, exc=True)
        value['users'][i] = user['_id']
    value['public'] = bool(value.get('public'))


@setting_utilities.validator(PluginSettings.HISTOMICSTK_QUARANTINE_FOLDER)
def validateHistomicsTKQuarantineFolder(doc):
    if not doc.get('value', None):
        doc['value'] = None
    else:
        Folder().load(doc['value'], force=True, exc=True)


# Defaults that have fixed values are added to the system defaults dictionary.
SettingDefault.defaults.update({
    PluginSettings.HISTOMICSTK_WEBROOT_PATH: 'histomicstk',
    PluginSettings.HISTOMICSTK_BRAND_NAME: 'HistomicsTK',
    PluginSettings.HISTOMICSTK_BANNER_COLOR: '#f8f8f8',
    PluginSettings.HISTOMICSTK_BRAND_COLOR: '#777777',
    PluginSettings.HISTOMICSTK_ANALYSIS_ACCESS: {'public': True},
})


class HistomicsTKResource(DockerResource):
    def __init__(self, name, *args, **kwargs):
        super(HistomicsTKResource, self).__init__(name, *args, **kwargs)
        self.route('GET', ('settings',), self.getPublicSettings)
        self.route('PUT', ('quarantine', ':id'), self.putQuarantine)
        self.route('PUT', ('quarantine', ':id', 'restore'), self.restoreQuarantine)
        self.route('GET', ('analysis', 'access'), self.getAnalysisAccess)

    def _accessList(self):
        access = Setting().get(PluginSettings.HISTOMICSTK_ANALYSIS_ACCESS) or {}
        acList = {
            'users': [{'id': x, 'level': AccessType.READ}
                      for x in access.get('users', [])],
            'groups': [{'id': x, 'level': AccessType.READ}
                       for x in access.get('groups', [])],
            'public': access.get('public', True),
        }
        for user in acList['users'][:]:
            userDoc = User().load(
                user['id'], force=True,
                fields=['firstName', 'lastName', 'login'])
            if userDoc is None:
                acList['users'].remove(user)
            else:
                user['login'] = userDoc['login']
                user['name'] = ' '.join((userDoc['firstName'], userDoc['lastName']))
        for grp in acList['groups'][:]:
            grpDoc = Group().load(
                grp['id'], force=True, fields=['name', 'description'])
            if grpDoc is None:
                acList['groups'].remove(grp)
            else:
                grp['name'] = grpDoc['name']
                grp['description'] = grpDoc['description']
        return acList

    @access.public
    @describeRoute(
        Description('List docker images and their CLIs')
    )
    def getDockerImages(self, *args, **kwargs):
        user = self.getCurrentUser()
        if not user:
            return {}
        result = super(HistomicsTKResource, self).getDockerImages(*args, **kwargs)
        acList = self._accessList()
        # Use the User().hasAccess to test for access via a synthetic document
        if not User().hasAccess({'access': acList, 'public': acList['public']}, user):
            return {}
        return result

    @describeRoute(
        Description('Get public settings for HistomicsTK.')
    )
    @access.public
    def getPublicSettings(self, params):
        keys = [
            PluginSettings.HISTOMICSTK_DEFAULT_DRAW_STYLES,
            PluginSettings.HISTOMICSTK_QUARANTINE_FOLDER,
        ]
        result = {k: self.model('setting').get(k) for k in keys}
        result[PluginSettings.HISTOMICSTK_QUARANTINE_FOLDER] = bool(
            result[PluginSettings.HISTOMICSTK_QUARANTINE_FOLDER])
        return result

    @describeRoute(
        Description('Get the access list for analyses.')
    )
    @access.admin
    def getAnalysisAccess(self, params):
        return self._accessList()

    @autoDescribeRoute(
        Description('Move an item to the quarantine folder.')
        .responseClass('Item')
        .modelParam('id', model=Item, level=AccessType.WRITE)
        .errorResponse('ID was invalid.')
        .errorResponse('Write access was denied for the item', 403)
    )
    @access.user(scope=TokenScope.DATA_WRITE)
    @filtermodel(model=Item)
    def putQuarantine(self, item):
        folder = Setting().get(PluginSettings.HISTOMICSTK_QUARANTINE_FOLDER)
        if not folder:
            raise RestException('The quarantine folder is not configured.')
        folder = Folder().load(folder, force=True, exc=True)
        if not folder:
            raise RestException('The quarantine folder does not exist.')
        if str(folder['_id']) == str(item['folderId']):
            raise RestException('The item is already in the quarantine folder.')
        originalFolder = Folder().load(item['folderId'], force=True)
        quarantineInfo = {
            'originalFolderId': item['folderId'],
            'originalBaseParentType': item['baseParentType'],
            'originalBaseParentId': item['baseParentId'],
            'originalUpdated': item['updated'],
            'quarantineUserId': self.getCurrentUser()['_id'],
            'quarantineTime': datetime.datetime.utcnow()
        }
        item = Item().move(item, folder)
        placeholder = Item().createItem(
            item['name'], {'_id': item['creatorId']}, originalFolder,
            description=item['description'])
        quarantineInfo['placeholderItemId'] = placeholder['_id']
        item.setdefault('meta', {})['quarantine'] = quarantineInfo
        item = Item().updateItem(item)
        placeholderInfo = {
            'quarantined': True,
            'quarantineTime': quarantineInfo['quarantineTime']
        }
        placeholder.setdefault('meta', {})['quarantine'] = placeholderInfo
        placeholder = Item().updateItem(placeholder)
        return item

    @autoDescribeRoute(
        Description('Restore a quarantined item to its original folder.')
        .responseClass('Item')
        .modelParam('id', model=Item, level=AccessType.WRITE)
        .errorResponse('ID was invalid.')
        .errorResponse('Write access was denied for the item', 403)
    )
    @access.admin
    @filtermodel(model=Item)
    def restoreQuarantine(self, item):
        if not item.get('meta', {}).get('quarantine'):
            raise RestException('The item has no quarantine record.')
        folder = Folder().load(item['meta']['quarantine']['originalFolderId'], force=True)
        if not folder:
            raise RestException('The original folder is not accesible.')
        placeholder = Item().load(item['meta']['quarantine']['placeholderItemId'], force=True)
        item = Item().move(item, folder)
        item['updated'] = item['meta']['quarantine']['originalUpdated']
        del item['meta']['quarantine']
        item = Item().updateItem(item)
        if placeholder is not None:
            Item().remove(placeholder)
        return item


def _saveJob(event):
    """
    When a job is saved, if it is a docker run task, add the Dask Bokeh port to
    the list of exposed ports.
    """
    job = event.info
    try:
        from bson import json_util

        jobkwargs = json_util.loads(job['kwargs'])
        if ('docker_run_args' not in jobkwargs['task'] and
                'scheduler' in jobkwargs['inputs']):
            jobkwargs['task']['docker_run_args'] = {'ports': {'8787': None}}
            job['kwargs'] = json_util.dumps(jobkwargs)
    except Exception:
        pass


class WebrootHistomicsTK(Webroot):
    def _renderHTML(self):
        self.updateHtmlVars({
            'title': Setting().get(PluginSettings.HISTOMICSTK_BRAND_NAME),
            'htkBrandName': Setting().get(PluginSettings.HISTOMICSTK_BRAND_NAME),
            'htkBrandColor': Setting().get(PluginSettings.HISTOMICSTK_BRAND_COLOR),
            'htkBannerColor': Setting().get(PluginSettings.HISTOMICSTK_BANNER_COLOR),
        })
        return super(WebrootHistomicsTK, self)._renderHTML()


def load(info):

    girderRoot = info['serverRoot']
    histomicsRoot = WebrootHistomicsTK(_template)
    histomicsRoot.updateHtmlVars(girderRoot.vars)

    ImageBrowseResource(info['apiRoot'])

    # The interface is always available under histomicstk and also available
    # under the specified path.
    info['serverRoot'].histomicstk = histomicsRoot
    webrootPath = Setting().get(PluginSettings.HISTOMICSTK_WEBROOT_PATH)
    setattr(info['serverRoot'], webrootPath, histomicsRoot)
    info['serverRoot'].girder = girderRoot

    pluginName = 'HistomicsTK'
    # create root resource for all REST end points of HistomicsTK
    resource = HistomicsTKResource(pluginName)
    setattr(info['apiRoot'], resource.resourceName, resource)

    # load docker images from cache
    dockerImageModel = ModelImporter.model('docker_image_model',
                                           'slicer_cli_web')
    dockerCache = dockerImageModel.loadAllImages()

    # generate REST end points for slicer CLIs of each docker image
    genRESTEndPointsForSlicerCLIsInDockerCache(resource, dockerCache)

    # auto-ingest annotations into database when a .anot file is uploaded
    events.bind('data.process', pluginName, process_annotations)

    events.bind('jobs.job.update.after', resource.resourceName,
                resource.AddRestEndpoints)

    events.bind('model.job.save', pluginName, _saveJob)

    def updateWebroot(event):
        """
        If the webroot path setting is changed, bind the new path to the
        histomicstk webroot resource.
        """
        if event.info.get('key') == PluginSettings.HISTOMICSTK_WEBROOT_PATH:
            setattr(info['serverRoot'], event.info['value'], histomicsRoot)

    events.bind('model.setting.save.after', 'histomicstk', updateWebroot)

    curConfig = config.getConfig().get('histomicstk', {})
    if curConfig.get('restrict_downloads'):
        # Change some endpoints to require token access
        endpoints = [
            ('collection', 'GET', (':id', 'download')),
            ('file', 'GET', (':id', 'download')),
            ('file', 'GET', (':id', 'download', ':name')),
            ('folder', 'GET', (':id', 'download')),
            ('item', 'GET', (':id', 'download')),
            ('resource', 'GET', ('download', )),
            ('resource', 'POST', ('download', )),

            ('item', 'GET', (':itemId', 'tiles', 'images', ':image')),
        ]

        for resource, method, route in endpoints:
            cls = getattr(info['apiRoot'], resource)
            func = cls.getRouteHandler(method, route)
            if func.accessLevel == 'public':
                func = access.token(func)
                cls.removeRoute(method, route)
                cls.route(method, route, func)
