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

    Job.updateJob(
        job,
        log='Started to Load Docker images\n',
        status=JobStatus.RUNNING,
        notify=False,
        progressMessage='caching docker image clis',
    )

    try:
        data = getDockerImages()
        if len(data) != 1:
            cachedData = data[0]

            for item in data:

                if not isinstance(item, dict):
                    imageCachedData={}
                    #print item
                    clis = subprocess.check_output(['docker', 'run', item,
                                                    '--list_cli'])
                    clis = json.loads(clis)
                    #print clis
                    for (dictKey, dictValue) in iteritems(clis):
                        print dictKey, dictValue
                        newCLI = {}

                        newCLI['type'] = dictValue['type']
                        xmlData = subprocess.check_output(['docker', 'run',item, dictKey,
                                                           '--xml'])
                        newCLI['xml'] = xmlData
                        #print xmlData

                        imageCachedData[dictKey] = newCLI
                    imageKey = hashlib.sha256(item.encode()).hexdigest()
                    imageCachedData['docker_image_name'] = item
                    cachedData[imageKey] = imageCachedData
            #print cachedData
            ModelImporter.model('setting').set(PluginSettings.DOCKER_IMAGES,
                                               [cachedData])
    except:
        raise("Bad exception")

    Job.updateJob(
        job,
        log='Finished caching docker xml\n',
        status=JobStatus.SUCCESS,
        notify=False,
        progressMessage='Completed caching docker images'
    )
    print("done job")
