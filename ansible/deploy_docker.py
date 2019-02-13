#!/usr/bin/env python

import argparse
import collections
import docker
import getpass
import gzip
import json
import os
import six
import socket
import sys
import tarfile
import time
import uuid
from distutils.version import LooseVersion

if not (LooseVersion('1.9') <= LooseVersion(docker.version)):
    raise Exception('docker or docker-py must be >= version 1.9')


BaseName = 'histomicstk'
ImageList = collections.OrderedDict([
    ('rmq', {
        'tag': 'rabbitmq:management',
        'name': 'histomicstk_rmq',
        'pull': True,
    }),
    ('mongodb', {
        'tag': 'mongo:latest',
        'name': 'histomicstk_mongodb',
        'pull': True,
    }),
    ('worker', {
        'tag': 'dsarchive/girder_worker',
        'name': 'histomicstk_girder_worker',
        'dockerfile': 'Dockerfile-girder-worker',
        'pinned': 'v0.1.5',
    }),
    ('histomicstk', {
        'tag': 'dsarchive/histomicstk_main',
        'name': 'histomicstk_histomicstk',
        'dockerfile': 'Dockerfile-histomicstk',
        'pinned': 'v0.1.5',
    }),
    ('cli', {
        'tag': 'dsarchive/histomicstk',
        'pull': True,
        'pinned': 'v0.1.6',
    }),
])


def config_mounts(mounts, config):
    """
    Add extra mounts to a docker configuration.

    :param mounts: a list of mounts to add, or None.
    :config: a config dictionary.  Mounts are added to the binds entry.
    """
    mountNumber = 1
    if mounts is None:
        mounts = []
    for mount in mounts:
        mountParts = mount.split(':')
        if len(mountParts) < 2:
            mountParts.append('')
        if mountParts[1] == '':
            mountParts[1] = 'mount%d' % mountNumber
            mountNumber += 1
        if '/' not in mountParts[1]:
            mountParts[1] = '/opt/histomicstk/mounts/%s' % mountParts[1]
        config['binds'].append(':'.join(mountParts))


def containers_provision(**kwargs):  # noqa
    """
    Provision or reprovision the containers.
    """
    client = docker_client()
    ctn = get_docker_image_and_container(
        client, 'histomicstk', version=kwargs.get('pinned'))

    if kwargs.get('conf'):
        merge_configuration(client, ctn, **kwargs)

    username = kwargs.get('username')
    password = kwargs.get('password')
    if username == '':
        username = six.moves.input('Admin login: ')
    if password == '':
        password = getpass.getpass('Password for %s: ' % (
            username if username else 'default admin user'))
    # docker exec -i -t histomicstk_histomicstk bash -c
    # 'cd /home/ubuntu/HistomicsTK/ansible && ansible-playbook -i
    # inventory/local docker_ansible.yml --extra-vars=docker=provision'
    extra_vars = {
        'docker': 'provision'
    }
    if username:
        extra_vars['girder_admin_user'] = username
        extra_vars['girder_no_create_admin'] = True
    if password:
        extra_vars['girder_admin_password'] = password
        extra_vars['girder_no_create_admin'] = True
    if kwargs.get('worker_api_url'):
        extra_vars['girder_api_url'] = kwargs['worker_api_url']
    if kwargs.get('cli'):
        extra_vars['cli_image'] = tag_with_version('cli', **kwargs)
        if kwargs['cli'] == 'test':
            extra_vars['cli_image_test'] = 'true'

    wait_for_girder(client, ctn)

    ansible_command = (
        'ansible-playbook -i inventory/local docker_ansible.yml '
        '--extra-vars=' + six.moves.shlex_quote(json.dumps(extra_vars)))
    exec_command = 'bash -c ' + six.moves.shlex_quote(
        'cd /home/ubuntu/HistomicsTK/ansible && ' + ansible_command)
    tries = 1
    while True:
        try:
            cmd = client.exec_create(
                container=ctn.get('Id'), cmd=exec_command, tty=True)
            try:
                for output in client.exec_start(cmd.get('Id'), stream=True):
                    print(convert_to_text(output).strip())
            except socket.error:
                pass
            cmd = client.exec_inspect(cmd.get('Id'))
            if not cmd['ExitCode']:
                break
        except (ValueError, docker.errors.APIError):
            time.sleep(1)
        print('Error provisioning (try %d)' % tries)
        tries += 1
        if not kwargs.get('retry'):
            raise Exception('Failed to provision')


def containers_start(port=8080, rmq='docker', mongo='docker', provision=False,
                     **kwargs):
    """
    Start all appropriate containers.  This is, at least, girder_worker and
    histomicstk.  Optionally, mongodb and rabbitmq are included.

    :param port: default port to expose.
    :param rmq: 'docker' to use a docker for rabbitmq, 'host' to use the docker
        host, otherwise the IP for the rabbitmq instance, where DOCKER_HOST
        maps to the docker host and anything else is passed through.
    :param mongo: 'docker' to use a docker for mongo, 'host' to use the docker
        host, otherwise the IP for the mongo instance, where DOCKER_HOST maps
        to the docker host and anything else is passed through.  The database
        is always 'girder'.  Any other value is considered a docker version.
    :param provision: if True, reprovision after starting.  Otherwise, only
        provision if the histomictk container is created.
    """
    client = docker_client()
    env = {
        'HOST_UID': os.popen('id -u').read().strip(),
        'HOST_GID': os.popen('id -g').read().strip(),
    }
    sockpath = '/var/run/docker.sock'
    if os.path.exists(sockpath):
        env['HOST_DOCKER_GID'] = str(os.stat(sockpath).st_gid)
    else:
        try:
            env['HOST_DOCKER_GID'] = os.popen('getent group docker').read().split(':')[2]
        except Exception:
            pass
    network_create(client, BaseName)

    for key in ImageList:
        func = 'container_start_' + key
        if func in globals():
            if globals()[func](client, env, key, port=port, rmq=rmq,
                               mongo=mongo, provision=provision, **kwargs):
                provision = True
    if provision:
        containers_provision(**kwargs)


def container_start_histomicstk(client, env, key='histomicstk', port=8080,
                                rmq='docker', mongo='docker', provision=False,
                                **kwargs):
    """
    Start a histomicstk container.

    :param client: docker client.
    :param env: dictionary to store environment variables.
    :param key: key within the ImageList.
    :param port: default port to expose.
    :param rmq: 'docker' to use a docker for rabbitmq, 'host' to use the docker
        host, otherwise the IP for the rabbitmq instance, where DOCKER_HOST
        maps to the docker host and anything else is passed through.
    :param mongo: 'docker' to use a docker for mongo, 'host' to use the docker
        host, otherwise the IP for the mongo instance, where DOCKER_HOST maps
        to the docker host and anything else is passed through.  The database
        is always 'girder'.  Any other value is considered a docker version.
    :param provision: if True, reprovision after starting.  Otherwise, only
        provision if the histomictk container is created.
    :returns: True if the container should be provisioned.
    """
    image = tag_with_version(key, **kwargs)
    name = ImageList[key]['name']
    ctn = get_docker_image_and_container(
        client, key, version=kwargs.get('pinned'))
    if ctn is None:
        provision = True
        config = {
            'restart_policy': {'name': 'always'},
            'privileged': True,  # so we can run docker
            'links': {},
            'port_bindings': {8080: int(port)},
            'binds': [
                get_path(kwargs['logs']) + ':/opt/logs:rw',
                get_path(kwargs['logs']) + ':/opt/histomicstk/logs:rw',
                get_path(kwargs['assetstore']) + ':/opt/histomicstk/assetstore:rw',
            ],
        }
        config['binds'].extend(docker_mounts())
        config_mounts(kwargs.get('mount'), config)
        if rmq == 'docker':
            config['links'][ImageList['rmq']['name']] = 'rmq'
        if mongo != 'host':
            config['links'][ImageList['mongodb']['name']] = 'mongodb'
        params = {
            'image': image,
            'detach': True,
            'hostname': key,
            'name': name,
            'environment': env.copy(),
            'ports': [8080],
        }
        print('Creating %s - %s' % (image, name))
        ctn = client.create_container(
            host_config=client.create_host_config(**config),
            networking_config=client.create_networking_config({
                BaseName: client.create_endpoint_config(aliases=[key])
            }),
            **params)
    if ctn.get('State') != 'running':
        print('Starting %s - %s' % (image, name))
    client.start(container=ctn.get('Id'))
    return provision


def container_start_mongodb(client, env, key='mongodb', mongo='docker',
                            mongodb_path='docker', **kwargs):
    """
    Start a mongo container if desired, or set an environment variable so other
    containers know where to find it.

    :param client: docker client.
    :param env: dictionary to store environment variables.
    :param key: key within the ImageList.
    :param mongo: 'docker' to use a docker for mongo, 'host' to use the docker
        host, otherwise the IP for the mongo instance, where DOCKER_HOST maps
        to the docker host and anything else is passed through.  The database
        is always 'girder'.  Any other value is considered a docker version.
    :param mongodb_path: the path to use for mongo when run in docker.  If
        'docker', use an internal data directory.
    """
    if mongo == 'host':
        env['HOST_MONGO'] = 'true'
        # If we generate the girder worker config file on the fly, update this
        # to something like:
        # env['HOST_MONGO'] = mongo if mongo != 'host' else 'DOCKER_HOST'
    else:
        version = None if mongo == 'docker' else mongo
        image = tag_with_version(key, version=version, **kwargs)
        name = ImageList[key]['name']
        ctn = get_docker_image_and_container(
            client, key, version=version if version else kwargs.get('pinned'))
        if ctn is None:
            config = {
                'restart_policy': {'name': 'always'},
            }
            params = {
                'image': image,
                'detach': True,
                'hostname': key,
                'name': name,
            }
            if mongodb_path != 'docker':
                params['volumes'] = ['/data/db']
                config['binds'] = [
                    get_path(mongodb_path) + ':/data/db:rw',
                ]
            print('Creating %s - %s' % (image, name))
            ctn = client.create_container(
                host_config=client.create_host_config(**config),
                networking_config=client.create_networking_config({
                    BaseName: client.create_endpoint_config(aliases=[key])
                }),
                **params)
        if ctn.get('State') != 'running':
            print('Starting %s - %s' % (image, name))
        client.start(container=ctn.get('Id'))


def container_start_rmq(client, env, key='rmq', rmq='docker', rmqport=None,
                        **kwargs):
    """
    Start a rabbitmq container if desired, or set an environment variable so
    other containers know where to find it.

    :param client: docker client.
    :param env: dictionary to store environment variables.
    :param key: key within the ImageList.
    :param rmq: 'docker' to use a docker for rabbitmq, 'host' to use the docker
        host, otherwise the IP for the rabbitmq instance, where DOCKER_HOST
        maps to the docker host and anything else is passed through.
    :param rmqport: if specified, docker RMQ port to expose
    """
    if rmq == 'docker':
        image = tag_with_version(key, **kwargs)
        name = ImageList[key]['name']
        ctn = get_docker_image_and_container(
            client, key, version=kwargs.get('pinned'))
        if ctn is None:
            config = {
                'restart_policy': {'name': 'always'},
            }
            params = {
                'image': image,
                'detach': True,
                'hostname': key,
                'name': name,
                # 'ports': [15672],  # for management access
            }
            if rmqport:
                params['ports'] = [5672]
                config['port_bindings'] = {5672: int(rmqport)}
            print('Creating %s - %s' % (image, name))
            ctn = client.create_container(
                host_config=client.create_host_config(**config),
                networking_config=client.create_networking_config({
                    BaseName: client.create_endpoint_config(aliases=[key])
                }),
                **params)
        if ctn.get('State') != 'running':
            print('Starting %s - %s' % (image, name))
        client.start(container=ctn.get('Id'))
    else:
        env['HOST_RMQ'] = 'true' if rmq == 'host' else rmq
        # If we generate the girder worker config file on the fly, update this
        # to something like:
        # env['HOST_RMQ'] = rmq if rmq != 'host' else 'DOCKER_HOST'


def container_start_worker(client, env, key='worker', rmq='docker', **kwargs):
    """
    Start a girder_worker container.

    :param client: docker client.
    :param env: dictionary to store environment variables.
    :param key: key within the ImageList.
    :param rmq: 'docker' to use a docker for rabbitmq, 'host' to use the docker
        host, otherwise the IP for the rabbitmq instance, where DOCKER_HOST
        maps to the docker host and anything else is passed through.
    """
    image = tag_with_version(key, **kwargs)
    name = ImageList[key]['name']
    ctn = get_docker_image_and_container(
        client, key, version=kwargs.get('pinned'))
    if ctn is None:
        worker_tmp_root = (
            kwargs['worker_tmp_root'] if kwargs['worker_tmp_root'] else '/tmp/girder_worker')
        config = {
            'restart_policy': {'name': 'always'},
            'privileged': True,  # so we can run docker
            'links': {},
            'binds': [
                get_path(kwargs['logs']) + ':/opt/logs:rw',
                '%s:%s' % (worker_tmp_root, worker_tmp_root),
                get_path(kwargs['assetstore']) + ':/opt/histomicstk/assetstore:rw',
            ]
        }
        config['binds'].extend(docker_mounts())
        config_mounts(kwargs.get('mount'), config)
        if rmq == 'docker':
            config['links'][ImageList['rmq']['name']] = 'rmq'
        else:
            env['HOST_RMQ'] = 'true' if rmq == 'host' else rmq
        env['GIRDER_WORKER_TMP_ROOT'] = worker_tmp_root
        if 'concurrency' in kwargs:
            env['GIRDER_WORKER_CONCURRENCY'] = kwargs['concurrency']
        params = {
            'image': image,
            'detach': True,
            'hostname': '%s_%s' % (key, str(uuid.uuid1(uuid.getnode(), 0))[24:]),
            'name': name,
            'environment': env.copy(),
        }
        print('Creating %s - %s' % (image, name))
        ctn = client.create_container(
            host_config=client.create_host_config(**config),
            networking_config=client.create_networking_config({
                BaseName: client.create_endpoint_config(aliases=[key])
            }),
            **params)
    if ctn.get('State') != 'running':
        print('Starting %s - %s' % (image, name))
    client.start(container=ctn.get('Id'))


def containers_status(**kwargs):
    """
    Report the status of any containers we are responsible for.
    """
    client = docker_client()

    keys = ImageList.keys()
    results = []
    for key in keys:
        if 'name' not in ImageList:
            continue
        ctn = get_docker_image_and_container(client, key, False)
        entry = {
            'key': key,
            'name': ImageList[key]['name'],
            'state': 'not created',
        }
        if ctn:
            entry['state'] = ctn.get('State', entry['state'])
            entry['status'] = ctn.get('Status')
        results.append(entry)
    print_table(results, collections.OrderedDict([
        ('name', 'Name'),
        ('state', 'State'),
        ('status', 'Status')]))


def containers_stop(remove=False, **kwargs):
    """
    Stop and optionally remove any containers we are responsible for.

    :param remove: True to remove the containers.  False to just stop them.
    """
    client = docker_client()
    keys = list(ImageList.keys())
    keys.reverse()
    for key in keys:
        ctn = get_docker_image_and_container(client, key, False)
        if ctn:
            if ctn.get('State') != 'exited':
                print('Stopping %s' % (key))
                client.stop(container=ctn.get('Id'))
            if remove:
                print('Removing %s' % (key))
                client.remove_container(container=ctn.get('Id'))

    if remove:
        network_remove(client, BaseName)


def convert_to_text(value):
    """
    Make sure a value is a text type in a Python version generic manner.

    :param value: a value that is either a text or binary string.
    :returns value: a text value.
    """
    if isinstance(value, six.binary_type):
        value = value.decode('utf8')
    if not isinstance(value, six.text_type):
        value = str(value)
    return value


def docker_client():
    """
    Return the current docker client in a manner that works with both the
    docker-py and docker modules.
    """
    try:
        client = docker.from_env(version='auto')
    except TypeError:
        # On older versions of docker-py (such as 1.9), version isn't a
        # parameter, so try without it
        client = docker.from_env()
    client = client if not hasattr(client, 'api') else client.api
    return client


def docker_mounts():
    """
    Return a list of mounts needed to work with the host's docker.

    :return: a list of volumes need to work with girder.
    """
    docker_executable = '/usr/bin/docker'
    if not os.path.exists(docker_executable):
        import shutil
        if not six.PY3:
            import shutilwhich  # noqa
        docker_executable = shutil.which('docker')
    mounts = [
        docker_executable + ':/usr/bin/docker',
        '/var/run/docker.sock:/var/run/docker.sock',
    ]
    return mounts


def get_docker_image_and_container(client, key, pullOrBuild=True, version=None):
    """
    Given a key from the docker ImageList, check if an image is present.  If
    not, pull it.  Check if an associated container exists and return
    information on it if so.

    :param client: docker client.
    :param key: key in the ImageList.
    :param pullOrBuild: if True, try to pull or build the image if it isn't
        present.  If 'pull', try to pull the image (not build), even if we
        already have it.
    :param version: if True, use the pinned version when pulling.  If a string,
        use that version.  Otherwise, don't specify a version (which defaults
        to latest).
    :returns: docker container or None.
    """
    if pullOrBuild:
        pull = False
        image = tag_with_version(key, version)
        try:
            client.inspect_image(image)
        except docker.errors.NotFound:
            pull = True
        if pull or pullOrBuild == 'pull':
            print('Pulling %s' % image)
            try:
                client.pull(image)
            except Exception:
                if pullOrBuild == 'pull':
                    raise
                if not ImageList[key].get('pull'):
                    images_build(True, key)
    name = ImageList[key].get('name')
    if name:
        containers = client.containers(all=True)
        ctn = [entry for entry in containers if name in
               [val.strip('/') for val in entry.get('Names', [])]]
        if len(ctn):
            return ctn[0]
    return None


def get_path(path):
    """
    Resolve a path to its realpath, creating a directory there if it doesn't
    exist.

    :param path: path to resolve and possibly create.
    :return: the resolved path.
    """
    path = os.path.realpath(os.path.expanduser(path))
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def images_build(retry=False, names=None):
    r"""
    Build necessary docker images from our dockerfiles.

    This is equivalent to running:
    docker build --force-rm --tag dsarchive/girder_worker \
           -f Dockerfile-girder-worker .
    docker build --force-rm --tag dsarchive/histomicstk_main \
           -f Dockerfile-histomicstk .

    :param retry: True to retry until success
    :param names: None to build all, otherwise a string or a list of strings of
        names to build.
    """
    basepath = os.path.dirname(os.path.realpath(__file__))
    client = docker_client()

    if names is None:
        names = ImageList.keys()
    elif isinstance(names, six.string_types):
        names = [names]
    for name in ImageList:
        if not ImageList[name].get('dockerfile') or name not in names:
            continue
        tries = 1
        while True:
            errored = False
            print('Building %s%s' % (
                name, '(try %d)' % tries if tries > 1 else ''))
            buildStatus = client.build(
                path=basepath,
                tag=ImageList[name]['tag'],
                rm=True,
                pull=True,
                forcerm=True,
                dockerfile=ImageList[name]['dockerfile'],
                decode=True,
            )
            for status in buildStatus:
                statusLine = status.get('status', status.get('stream', '')).strip()
                try:
                    print(statusLine)
                except Exception:
                    print(repr(statusLine))
                if 'errorDetail' in status:
                    if not retry:
                        sys.exit(1)
                    errored = True
                    break
            if not errored:
                break
            print('Error building %s\n' % name)
            tries += 1
        print('Done building %s\n' % name)


def images_repull(**kwargs):
    """
    Repull all docker images.
    """
    client = docker_client()
    for key, image in six.iteritems(ImageList):
        if 'name' not in image and not kwargs.get('cli'):
            continue
        get_docker_image_and_container(
            client, key, 'pull',  version=kwargs.get('pinned'))


def merge_configuration(client, ctn, conf, **kwargs):
    """
    Merge a Girder configuration file with the one in a running container.

    :param client: the docker client.
    :param ctn: a running docker container that contains
        /opt/histomicstk/girder/girder/conf/girder.local.cfg
    :param conf: a path to a configuration file fragment to merge with the
        extant file.
    """
    cfgPath = '/opt/histomicstk/girder/girder/conf'
    cfgName = 'girder.local.cfg'
    tarStream, stat = client.get_archive(ctn, cfgPath + '/' + cfgName)
    if hasattr(tarStream, 'read'):
        tarData = tarStream.read()
    else:
        tarData = b''.join([part for part in tarStream])
    # Check if this is actually gzipped and uncompress it
    if tarData[:2] == b'\x1f\x8b':
        tarData = gzip.GzipFile(fileobj=six.BytesIO(tarData)).read()
    tarStream = six.BytesIO(tarData)
    tar = tarfile.TarFile(mode='r', fileobj=tarStream)
    parser = six.moves.configparser.SafeConfigParser()
    cfgFile = six.StringIO(convert_to_text(tar.extractfile(cfgName).read()))
    parser.readfp(cfgFile)
    parser.read(conf)
    output = six.StringIO()
    parser.write(output)
    output = output.getvalue()
    if kwargs.get('verbose') >= 1:
        print(output)
    if isinstance(output, six.text_type):
        output = output.encode('utf8')
    output = six.BytesIO(output)
    output.seek(0, os.SEEK_END)
    outputlen = output.tell()
    output.seek(0)
    tarOutput = six.BytesIO()
    tar = tarfile.TarFile(fileobj=tarOutput, mode='w')
    tarinfo = tarfile.TarInfo(name=cfgName)
    tarinfo.size = outputlen
    tarinfo.mtime = time.time()
    tar.addfile(tarinfo, output)
    tar.close()
    tarOutput.seek(0)
    client.put_archive(ctn, cfgPath, data=tarOutput)


def network_create(client, name):
    """
    Ensure a network exists with a specified name.

    :param client: docker client.
    :param name: name of the network.
    """
    networks = client.networks()
    net = [entry for entry in networks if name == entry.get('Name')]
    if len(net):
        return
    client.create_network(name)


def network_remove(client, name):
    """
    Ensure a network with a specified name is removed.

    :param client: docker client.
    :param name: name of the network.
    """
    networks = client.networks()
    net = [entry for entry in networks if name == entry.get('Name')]
    if not len(net):
        return
    client.remove_network(net[0].get('Id'))


def pinned_versions():
    """
    Get a list of images that have pinned versions.

    :return: a list of image names with versions.
    """
    pinned = []
    for image in six.itervalues(ImageList):
        if 'pinned' in image:
            pinned.append('%s:%s' % (image['tag'], image['pinned']))
    return pinned


def print_table(table, headers):
    """
    Format and print a table.

    :param table: a list of dictionaries to display.
    :param headers: an order dictionary of keys to display with the values
        being the column headers.
    """
    widths = {}
    for key in headers:
        widths[key] = len(str(headers[key]))
        for row in table:
            if key in row:
                widths[key] = max(widths[key], len(str(row[key])))
    format = '  '.join(['%%-%ds' % widths[key] for key in headers])
    print(format % tuple([headers[key] for key in headers]))
    for row in table:
        print(format % tuple([row.get(key, '') for key in headers]))


def show_info():
    """
    Print additional installation notes.
    """
    print("""
Running containers can be joined using a command like
  docker exec -i -t histomicstk_histomicstk bash

To allow docker containers to use memcached, make sure the host is running
memcached and it is listening on the docker IP address (or listening on all
addresses via -l 0.0.0.0).

To determine the current mongo docker version, use a command like
  docker exec histomicstk_mongodb mongo girder --eval 'db.version()'
To check if mongo can be upgrade, query the compatability mode via
  docker exec histomicstk_mongodb mongo girder --eval \\
  'db.adminCommand({getParameter: 1, featureCompatibilityVersion: 1})'
Mongo can only be upgraded if the compatibility version is the same as the
semi-major version.  Before upgrading, set the compatibility mode.  For
instance, if Mongo 3.6.1 is running,
  docker exec histomicstk_mongodb mongo girder --eval \\
  'db.adminCommand({setFeatureCompatibilityVersion: "3.6"})'
after which Mongo can be upgraded to version 4.  After upgrading, set the
compatibility mode to the new version.
""")


def tag_with_version(key, version=None, **kwargs):
    """
    Get an image tag with a version appended to it.  If the pinned parameter is
    specified, use the specified or pinned version.

    :param key: the key in the image list.
    :param version: the version to use, True to use the pinned value, or None
        to use latest.  If None, use kwargs.get('pinned') as the version.
    :return: the image tag with a version.
    """
    image = ImageList[key]['tag']
    if version is None:
        version = kwargs.get('pinned')
    if version is True:
        version = ImageList[key].get('pinned')
    if isinstance(version, six.string_types):
        image = image.split(':', 1)[0] + ':' + version
    if ':' not in image:
        image += ':latest'
    return image


def wait_for_girder(client, ctn, maxWait=3600):
    """
    Wait for Girder in a specific container to respond with its current
    version.

    :param client: docker client.
    :param ctn: docker container with Girder.
    :param maxWait: maximum time to wait for Girder to respond.
    """
    starttime = time.time()
    sys.stdout.write('Waiting for Girder to report version: ')
    sys.stdout.flush()
    # This really should be the girder_api_url from the current settings
    girder_api_url = 'http://histomicstk:8080/api/v1'
    exec_command = 'bash -c ' + six.moves.shlex_quote(
        'curl "%s/system/version"' % girder_api_url)
    while time.time() - starttime < maxWait:
        cmd = client.exec_create(
            container=ctn.get('Id'), cmd=exec_command, tty=True)
        output = client.exec_start(cmd.get('Id'), stream=False)
        try:
            output = json.loads(convert_to_text(output).strip())
            if 'apiVersion' in output:
                break
        except Exception:
            pass
        output = None
        time.sleep(1)
        sys.stdout.write('.')
        sys.stdout.flush()
    if not output:
        raise Exception('Girder never responded')
    sys.stdout.write(' %s\n' % output['apiVersion'])
    sys.stdout.write('Took {} seconds\n'.format(time.time() - starttime))


if __name__ == '__main__':   # noqa
    parser = argparse.ArgumentParser(
        description='Provision and run HistomicsTK in docker containers.')
    parser.add_argument(
        'command',
        choices=['start', 'restart', 'stop', 'rm', 'remove', 'status',
                 'build', 'provision', 'info', 'pull'],
        help='Start, stop, stop and remove, restart, check the status of, or '
             'build our own docker containers')
    parser.add_argument(
        '--assetstore', '-a', default='~/.histomicstk/assetstore',
        help='Assetstore path.')
    parser.add_argument(
        '--build', '-b', dest='build', action='store_true',
        help='Build gider_worker and histomicstk docker images.')
    parser.add_argument(
        '--cli', '-c', dest='cli', action='store_true', default=True,
        help='Pull and install the HistomicsTK cli docker image.')
    parser.add_argument(
        '--cli-test', dest='cli', action='store_const', const='test',
        help='Pull and install the HistomicsTK cli docker image; test the CLI.')
    parser.add_argument(
        '--no-cli', dest='cli', action='store_false',
        help='Do not pull or install the HistomicsTK cli docker image.')
    parser.add_argument(
        '--concurrency', '-j', type=int,
        help='Girder worker concurrency.')
    parser.add_argument(
        '--conf', '--cfg', '--girder-cfg',
        help='Merge a Girder configuration file with the default '
        'configuration in the docker container during provisioning.')
    parser.add_argument(
        '--db', '-d', dest='mongodb_path', default='~/.histomicstk/db',
        help='Database path (if a Mongo docker container is used).  Use '
             '"docker" for the default docker storage location.')
    parser.add_argument(
        '--image', action='append',
        help='Override docker image information.  The value is of the form '
        'key:tag:dockerfile.')
    parser.add_argument(
        '--info', action='store_true',
        help='Show installation and usage notes.')
    parser.add_argument(
        '--logs', '--log', '-l', default='~/.histomicstk/logs',
        help='Logs path.')
    parser.add_argument(
        '--mongo', '-m', default='docker',
        choices=['docker', 'host', '3.4', '3.6', '4.0', 'latest'],
        help='Either use mongo from docker or from host.  If a version is '
        'specified, the docker with that version will be used.')
    parser.add_argument(
        '--mount', '--extra', '-e', action='append',
        help='Extra volumes to mount.  These are mounted internally at '
        '/opt/histomicstk/mounts/(name), and are specified in the form '
        '(host path)[:(name)[:ro]].  If no name is specified, mountX is used, '
        'starting at mount1.')
    parser.add_argument(
        '--only', '-o',
        help='A comma separated list to only start specified containers.  '
        'Defaults to all containers (%s).' % (','.join([
            key for key in ImageList.keys() if key != 'cli']))),
    parser.add_argument(
        '--password', '--pass', '--passwd', '--pw',
        const='', default=None, nargs='?',
        help='Override the Girder admin password used in provisioning.  Set '
        'to an empty string to be prompted for username and password.')
    parser.add_argument(
        '--pinned', dest='pinned', action='store_true', default=False,
        help='When pulling images, use the pinned versions (%s).' % (
            ', '.join(pinned_versions())))
    parser.add_argument(
        '--latest', dest='pinned', action='store_false',
        help='When pulling images, use the latest images.')
    parser.add_argument(
        '--provision', action='store_true',
        help='Reprovision the Girder the docker containers are started.')
    parser.add_argument(
        '--port', '-p', type=int, default=8080,
        help='Girder access port.')
    parser.add_argument(
        '--pull', action='store_true',
        help='Repull docker images.')
    parser.add_argument(
        '--retry', '-r', action='store_true', default=True,
        help='Retry builds and provisioning until they succeed')
    parser.add_argument(
        '--rmqport', type=int,
        help='RabbitMQ access port (commonly 5672).')
    parser.add_argument(
        '--no-retry', '--once', '-1', dest='retry', action='store_false',
        help='Do not retry builds and provisioning until they succeed')
    parser.add_argument(
        '--rmq', default='docker',
        help='Either use rabbitmq from docker or from host (docker, host, or '
        'IP adress or hostname of host.')
    parser.add_argument(
        '--status', '-s', action='store_true',
        help='Report the status of relevant docker containers and images.')
    parser.add_argument(
        '--username', '--user', const='', default=None, nargs='?',
        help='Override the Girder admin username used in provisioning.  Set '
        'to an empty string to be prompted for username and password.')
    parser.add_argument(
        '--worker-api-url',
        help='The alternate Girder API URL used by workers to reach Girder.  '
        'This defaults to http://histomicstk:8080/api/v1')
    parser.add_argument(
        '--worker-tmp-root', '--tmp', default='/tmp/girder_worker',
        help='The path to use for the girder_worker tmp_root.  This must be '
        'reachable by the HistomicsTK and the girder_worker docker '
        'containers.  It cannot be a top-level directory.')
    parser.add_argument('--verbose', '-v', action='count', default=0)

    # Should we add an optional url or host value for rmq and mongo?
    # Should we allow installing git repos in a local directory to make it
    #   easier to develop python and javascript?
    # We should figure out how to run the ctests
    # Add a provisioning step to copy sample data (possibly by mounting the
    #   appropriate host directory if it exists).

    args = parser.parse_args()
    if args.verbose >= 2:
        print('Parsed arguments: %r' % args)

    if args.image:
        for imagestr in args.image:
            key, tag, dockerfile = imagestr.split(':')
            ImageList[key]['tag'] = tag
            ImageList[key]['dockerfile'] = dockerfile

    if args.only:
        only = args.only.split(',')
        for key in ImageList.keys():
            if key not in only:
                del ImageList[key]

    if args.info or args.command == 'info':
        show_info()

    if args.command == 'provision':
        args.command = 'start'
        args.provision = True

    if args.pull or args.command == 'pull':
        images_repull(**vars(args))
    if args.build or args.command == 'build':
        images_build(args.retry)
    if args.command in ('stop', 'restart', 'rm', 'remove'):
        containers_stop(remove=args.command in ('rm', 'remove'))
    if args.command in ('start', 'restart'):
        containers_start(**vars(args))
    if args.command in ('status', ) or args.status:
        containers_status(**vars(args))
