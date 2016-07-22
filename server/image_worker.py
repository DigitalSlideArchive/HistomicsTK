from six import iteritems
from girder.utility.model_importer import ModelImporter
from girder.plugins.jobs.constants import JobStatus
from .docker_resource import getDockerImages
from .constants import PluginSettings
import subprocess
import json
import hashlib

# TODO apply try catch blocks individually and catch specific exceptions
# TODO pull non existent image names


def loadXML(job):
    print('in worker')
    #print(job)
    Job = ModelImporter.model('job', 'jobs')
    name = job['kwargs']['name']
    Job.updateJob(
        job,
        log='Started to Load Docker images\n',
        status=JobStatus.RUNNING,
        notify=True,
        progressMessage='caching docker image clis',
    )

    try:
        cachedData = getDockerImages()
        print('cachedData %s'% type(cachedData))
        imageCachedData={}
        #print item
        clis = subprocess.check_output(['docker', 'run', name,
                                        '--list_cli'])
        clis = json.loads(clis)
        #print clis
        for (dictKey, dictValue) in iteritems(clis):
            print dictKey, dictValue
            newCLI = {}

            newCLI['type'] = dictValue['type']
            xmlData = subprocess.check_output(['docker', 'run',name, dictKey,
                                               '--xml'])
            newCLI['xml'] = xmlData
            #print xmlData
            Job.updateJob(
                job,
                log='Started to Load Docker images\n',
                status=JobStatus.RUNNING,
                notify=True,
                progressMessage='caching clis %s' % dictKey,
            )
            imageCachedData[dictKey] = newCLI
            print('update message')
        imageKey = hashlib.sha256(name.encode()).hexdigest()

        imageCachedData['docker_image_name'] = name
        #print imageCachedData
        cachedData[imageKey] = imageCachedData
        print cachedData

        ModelImporter.model('setting').set(PluginSettings.DOCKER_IMAGES,cachedData)
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
