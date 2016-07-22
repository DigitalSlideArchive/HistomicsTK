from six import iteritems
from girder.utility.model_importer import ModelImporter
from girder.plugins.jobs.constants import JobStatus
from .docker_resource import getDockerImages,DockerCache
from .constants import PluginSettings
import subprocess
import json
import hashlib

# TODO apply try catch blocks individually and catch specific exceptions
# TODO pull non existent image names


def loadXML(job):
    print('in worker')

    Job = ModelImporter.model('job', 'jobs')
    names = job['kwargs']['name']

    Job.updateJob(
        job,
        log='Started to Load Docker images\n',
        status=JobStatus.RUNNING,
        notify=True,
        progressMessage='caching docker image clis',
    )

    try:
        cachedData = DockerCache(getDockerImages())

        for name in names:



            clis = subprocess.check_output(['docker', 'run', name,
                                            '--list_cli'])
            cachedData.addImage(name)
            clis = json.loads(clis)

            for (dictKey, dictValue) in iteritems(clis):


                xmlData = subprocess.check_output(['docker', 'run', name, dictKey,
                                                   '--xml'])

                cachedData.addCLI(name, dictKey, dictValue['type'], xmlData)

                Job.updateJob(
                    job,
                    log='Started to Load Docker images\n',

                    notify=True,
                    progressMessage='caching cli %s from image %s' % (dictKey,name),
                )

        ModelImporter.model('setting').set(PluginSettings.DOCKER_IMAGES, cachedData.raw())
    except:
        raise("Bad exception")

    Job.updateJob(
        job,
        log='Finished caching docker xml\n',
        status=JobStatus.SUCCESS,
        notify=True,
        progressMessage='Completed docker images %s'% name
    )
    print("done job")
