#!/usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################################
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
##############################################################################


from six import iteritems
import json

from girder.utility.model_importer import ModelImporter
from girder.plugins.jobs.constants import JobStatus
from .docker_resource import getDockerImageSettings, DockerCache, \
    DockerResource, DockerImageError
from .rest_slicer_cli import getDockerImage, getDockerImageCLIXMLSpec, \
    getDockerImageCLIList
from .constants import PluginSettings


def loadXML(job):
    Job = ModelImporter.model('job', 'jobs')
    names = job['kwargs']['name']
    oldSettings = DockerCache(job['kwargs']['oldSettings'])

    Job.updateJob(
        job,
        log='Started to Load Docker images\n',
        status=JobStatus.RUNNING,
        notify=True,
        progressMessage='caching docker image clis',
    )

    try:
        cachedData = DockerCache({})

        for name in names:

            getDockerImage(name, True)

            clis = getDockerImageCLIList(name)

            cachedData.addImage(name)

            try:
                clis = json.loads(clis)
            except ValueError as err:
                raise DockerImageError(err.message +
                                       '\ncli list format is incorrect', name)

            for (dictKey, dictValue) in iteritems(clis):
                xmlData = getDockerImageCLIXMLSpec(name, dictKey)

                cachedData.addCLI(name, dictKey, dictValue['type'], xmlData)

                Job.updateJob(
                    job,
                    log='Loaded cli %s\n' % dictKey,

                    notify=True,
                    progressMessage='caching cli %s from '
                                    'image %s' % (dictKey, name),
                )

        if saveSetting(oldSettings, cachedData):
            Job.updateJob(
                job,
                log='Finished caching docker xml\n',
                status=JobStatus.SUCCESS,
                notify=True,
                progressMessage='Completed caching '
                                'docker images %s' % name
            )

            return
        else:
            Job.updateJob(
                job,
                log='Failed to cache docker image data, setting was modified '
                    'since the job started\n',
                status=JobStatus.ERROR,
                notify=True,
                progressMessage='Failed to cache docker images %s' % name
            )
            # rerun job since settings were modifies since job ran
            DockerResource.appendImageJob(names)
            return
    except Exception as err:
        print "exception"
        Job.updateJob(
            job,
            log='Failed to cache docker image data %s \n' % err.message,
            status=JobStatus.ERROR,
            notify=True,
            progressMessage='Failed to cache docker images %s' % name
        )


def verifyDictionary(job):
    Job = ModelImporter.model('job', 'jobs')
    newSettings = DockerCache(job['kwargs']['newSettings'])
    oldSettings = DockerCache(job['kwargs']['oldSettings'])
    # validate the structure of the new settings
    Job.updateJob(
        job,
        log='Started to verfy the settings\n',
        status=JobStatus.RUNNING,
        notify=True,
        progressMessage='Started to verfy the settings\n',
    )
    try:

        newSettings.validate()

        for img in newSettings.getDockerImageList():
            getDockerImage(img, True)

            cli_string = getDockerImageCLIList(img)

            try:
                cli_dict = json.loads(cli_string)
            except ValueError as err:
                raise DockerImageError(err.message +
                                       '\ncli list format is incorrect', img)

            for (cli, type) in iteritems(newSettings.getCLIListSpec(img)):
                if cli not in cli_dict:
                    raise DockerImageError(
                        'The cli %s does not exist in the '
                        'image %s. ' % (cli, img), img)
                if type['type'] != cli_dict[cli]['type']:
                    raise DockerImageError(
                        'The cli %s does not have the appropriate '
                        'type %s %s' % (cli_dict[cli]['type'], type['type']),
                        img)
                xml = getDockerImageCLIXMLSpec(img, cli)

                if xml != newSettings.getCLIXML(img, cli):
                    raise DockerImageError(
                        'The xml spec of image %s cli %s does not match.' % (
                            img, cli), img)

        if saveSetting(oldSettings, newSettings):
            Job.updateJob(
                job,
                log='Finished caching setting\n',
                status=JobStatus.SUCCESS,
                notify=True,
                progressMessage='Completed caching setting'
            )
            return
        else:
            Job.updateJob(
                job,
                log='The original setting changed since the job finished '
                    'must rerun the job \n',
                status=JobStatus.ERROR,
                notify=True,
                progressMessage='The original setting changed since the job '
                                'finished must rerun the job'
            )
            DockerResource.validateDict(newSettings)

    except Exception as err:

        Job.updateJob(
            job,
            log='Failed to cache docker setting %s \n' % err.message,
            status=JobStatus.ERROR,
            notify=True,
            progressMessage='Failed to use new docker setting'
        )


def saveSetting(oldSettings, newSettings):
    currentSettings = DockerCache(getDockerImageSettings())

    if oldSettings.equals(currentSettings):
        newSettingSave = ModelImporter.model('setting').findOne(
            {'key': PluginSettings.DOCKER_IMAGES})

        if newSettingSave is None:
            newSettingSave = {}
        else:
            newSettingSave['key'] = PluginSettings.DOCKER_IMAGES
            newSettingSave['value'] = newSettings.getRawData()

        ModelImporter.model('setting').save(newSettingSave, validate=False)
        return True
    else:
        return False
