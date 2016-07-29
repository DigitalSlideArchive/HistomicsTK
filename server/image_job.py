from docker import Client
from docker.errors import DockerException


from girder.models.model_base import ModelImporter
from girder.plugins.jobs.constants import JobStatus
import json
from server.models.docker_image import DockerImage, DockerImageError, \
    DockerImageNotFoundError, DockerCache
from six import iteritems


def deleteImage(job):
    jobModel = ModelImporter.model('job', 'jobs')

    jobModel.updateJob(
        job,
        log='Started to Delete Docker images\n',
        status=JobStatus.RUNNING,
    )

    deleteList = job['kwargs']['deleteList']
    error = False

    try:
        docker_client = Client(base_url='unix://var/run/docker.sock')

    except DockerException as err:
        jobModel.updateJob(
            job,
            log='Failed to create the Docker Client\n' + err.__str__() + '\n',
            status=JobStatus.ERROR,
        )
        raise DockerImageError('Could not create the docker client')

    for name in deleteList:
        try:
            docker_client.remove_image(name, force=True)
        except Exception as err:
            jobModel.updateJob(
                job,
                log='Failed to remove %s image \n' % name +
                    err.__str__() + '\n',
                status=JobStatus.RUNNING,
            )
            error = True
    if error is True:
        jobModel.updateJob(
            job,
            log='Failed to remove some images',
            status=JobStatus.ERROR,
            notify=True,
            progressMessage='Errors deleting some images'
        )
    else:
        jobModel.updateJob(
            job,
            log='Removed all images',
            status=JobStatus.SUCCESS,
            notify=True,
            progressMessage='Removed all images'
        )


def jobPullAndLoad(job):
    try:
        jobModel = ModelImporter.model('job', 'jobs')
        pullList = job['kwargs']['pullList']
        loadList = job['kwargs']['loadList']
        notExistList = []
        notExistSet = set()
        jobModel.updateJob(
            job,
            log='Started to Load Docker images\n',
            status=JobStatus.RUNNING,
        )
        try:
            docker_client = Client(base_url='unix://var/run/docker.sock')

        except DockerException as err:
            jobModel.updateJob(
                job,
                log='Failed to create the Docker Client\n' + err.__str__()+'\n',
                status=JobStatus.ERROR,
            )
            raise DockerImageError('Could not create the docker client')

        try:

            pullDockerImage(docker_client, pullList)

        except DockerImageNotFoundError as err:

            notExistList = err.imageName
            notExistSet = set(err.imageName)
            jobModel.updateJob(
                job,
                log='could not find the following '
                    'images\n'+'\n'.join(notExistList)+'\n',
                status=JobStatus.ERROR,
            )

        cache = DockerCache()
        for name in pullList:
            if name not in notExistSet:
                jobModel.updateJob(
                    job,
                    log='Image %s was pulled successfully \n' % name,
                    status=JobStatus.RUNNING,
                )
                # create dictionary and load to database

                try:
                    dockerImg = DockerImage(name)
                    getCliData(name, docker_client, dockerImg, jobModel, job)
                    cache.addImage(dockerImg)
                    jobModel.updateJob(
                        job,
                        log='Got pulled image %s meta data \n' % name,
                        status=JobStatus.RUNNING,
                    )
                except DockerImageError as err:
                    jobModel.updateJob(
                        job,
                        log='Error with recently'
                            ' pulled image %s' % name + err.__str__()+'\n',
                        status=JobStatus.ERROR,

                    )

        for name in loadList:
            # create dictionary and load to database
            try:
                dockerImg = DockerImage(name)
                getCliData(name, docker_client, dockerImg, jobModel, job)
                cache.addImage(dockerImg)
                jobModel.updateJob(
                    job,
                    log='Loaded meta data from pre-existing local'
                        'image %s\n' % name,
                    status=JobStatus.ERROR,

                )
            except DockerImageError as err:
                jobModel.updateJob(
                    job,
                    log='Error with recently loading pre-existing image'
                        'image %s \n ' % name + err.__str__()+'\n',
                    status=JobStatus.ERROR,

                )

        imageModel = ModelImporter.model('dockerimagemodel', 'HistomicsTK')

        imageModel.saveAllImgs(cache)

        jobModel.updateJob(
            job,
            log='Finished caching Docker image data\n',
            status=JobStatus.SUCCESS,
            notify=True,
            progressMessage='Completed caching setting'
        )
    except Exception as err:
        jobModel.updateJob(
            job,
            log='Error with job'
                '\n %s \n ' % name + err.__str__()+'\n',
            status=JobStatus.ERROR,

        )


def getDockerOutput(imgName, command, client):
    try:

        cont = client.create_container(image=imgName, command=command)
        client.start(container=cont.get('Id'))
        logs = client.logs(container=cont.get('Id'),
                           stdout=True, stderr=False, stream=True)
        ret_code = client.wait(container=cont.get('Id'))
    except Exception as err:
        raise DockerImageError(
            'Attempt to docker run %s %s failed'
            ' ' % (imgName, command) + err.__str__(), imgName)
    if ret_code != 0:
        raise DockerImageError(
            'Attempt to docker run %s %s failed' % (imgName, command), imgName)
    return "".join(logs)


def getCliData(name, client, img, jobModel, job):
    try:

        if isinstance(client, Client) and isinstance(img, DockerImage):

            cli_dict = getDockerOutput(name, '--list_cli', client)
            # contains nested dict
            """
            {<cliname>:{
                        type:<type>
                        }
            }
            """
            cli_dict = json.loads(cli_dict)

            for (key, val) in iteritems(cli_dict):

                cli_xml = getDockerOutput(name, '%s --xml' % key, client)
                cli_dict[key][DockerImage.xml] = cli_xml
                jobModel.updateJob(
                    job,
                    log='Got image %s,cli %s meta data'
                        ' \n' % (name, key),
                    status=JobStatus.RUNNING,
                )
                img.addCLI(key, cli_dict[key])
        return cli_dict
    except Exception as err:
        print err
        raise DockerImageError('Error getting %s cli '
                               'data from image %s'
                               ' ' % (name, img)+err.__str__())


def pullDockerImage(client, names):

    imgNotExistList = []
    for name in names:
        try:
            # (repo, tag) = parseName(name)
            client.pull(name)
        except DockerException:
            imgNotExistList.append(name)
    if len(imgNotExistList) != 0:

        raise DockerImageNotFoundError('Could not find multiple images ',
                                       image_name=imgNotExistList)


def parseName(name):

    if '@' in name:
        return name.split('@')
    elif ':' in name:
        return name.split(':')
    else:
        raise DockerImageError('The image name %s is in an incorrect format'
                               ' a digest or tag is required' % name)
