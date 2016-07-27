from six import iteritems
from girder.utility.model_importer import ModelImporter
from girder.plugins.jobs.constants import JobStatus
from .docker_resource import getDockerImageSettings, DockerCache, \
    DockerResource, localDockerImageCLIList, localDockerImageclixml, \
    localDockerImageExists
from .constants import PluginSettings
import subprocess
import json
import hashlib
import sys
import traceback


# TODO apply try catch blocks individually and catch specific exceptions
# TODO pull non existent image names


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
            id = localDockerImageExists(name)
            if id is None:
                raise ValueError('The image %s does not exist locally' % name)
            clis = subprocess.check_output(['docker', 'run', name,
                                            '--list_cli'])
            cachedData.addImage(name)
            clis = json.loads(clis)

            for (dictKey, dictValue) in iteritems(clis):
                xmlData = subprocess.check_output(
                    ['docker', 'run', name, dictKey,
                     '--xml'])

                cachedData.addCLI(name, dictKey, dictValue['type'], xmlData)

                Job.updateJob(
                    job,
                    log='Loaded cli %s\n' % dictKey,

                    notify=True,
                    progressMessage='caching cli %s from '
                                    'image %s' % (dictKey, name),
                )
        currentSettings = DockerCache(getDockerImageSettings())
        if currentSettings.equals(oldSettings):

            newSetting = ModelImporter.model('setting').findOne(
                {'key': PluginSettings.DOCKER_IMAGES})
            if newSetting is None:
                newSetting = {}
            else:

                newSetting['key'] = PluginSettings.DOCKER_IMAGES
                newSetting['value'] = cachedData.raw()
            ModelImporter.model('setting').save(newSetting, validate=False)

            Job.updateJob(
                job,
                log='Finished caching docker xml\n',
                status=JobStatus.SUCCESS,
                notify=True,
                progressMessage='Completed caching docker images %s' % name
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
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        errorInfo = traceback.format_exception(exc_type, exc_value,
                                               exc_traceback)
        Job.updateJob(
            job,
            log='Failed to cache docker image data %s \n' % errorInfo,
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
        try:
            newSettings.validate()
        except ValueError as err:
            raise ValueError(
                'The structure of the new settings is not correct \n%s' % err)
        for img in newSettings.getDockerImageList():
            id = localDockerImageExists(img)
            if id is None:
                raise ValueError('The image %s does not exist locally' % img)
            cli_string = localDockerImageCLIList(img)
            if cli_string is None:
                raise ValueError('clis donot exist for the image %s' % img)
            cli_dict = json.loads(cli_string)
            for (cli, type) in iteritems(newSettings.getCLIDict(img)):
                if cli not in cli_dict:
                    raise ValueError(
                        'The cli %s does not exist in the image. ' % cli)
                if type['type'] != cli_dict[cli]['type']:
                    raise ValueError(
                        'The cli %s does not have the appropriate '
                        'type %s %s' % (cli_dict[cli]['type'], type['type']))
                xml = localDockerImageclixml(img, cli)
                if xml is None:
                    raise ValueError(
                        'Could not get the xml spec of image %s cli %s.' % (
                            img, cli))
                if xml != newSettings.getCLIXML(img, cli):
                    raise ValueError(
                        'The xml spec of image %s cli %s does not match.' % (
                            img, cli))
        currentSettings = DockerCache(getDockerImageSettings())
        if oldSettings.equals(currentSettings):
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

    except ValueError as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        errorInfo = traceback.format_exception(exc_type, exc_value,
                                               exc_traceback)
        Job.updateJob(
            job,
            log='Failed to cache docker setting %s \n' % errorInfo,
            status=JobStatus.ERROR,
            notify=True,
            progressMessage='Failed to use new docker setting'
        )
