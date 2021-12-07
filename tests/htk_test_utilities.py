import os
import subprocess
import tempfile
import time
import warnings

import docker
import girder_client
import pytest


def getTestFilePath(name):
    """
    Return the path to a file in the tests/test_files directory.
    :param name: The name of the file.
    :returns: the path to the file.
    """
    return os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'test_files', name)


def _get_htk_ipaddr(dclient):
    # search docker containers for a DSA docker container
    # and fetch its IP address
    return list(dclient.containers.list(
        filters={'label': 'HISTOMICSTK_GC_TEST'})[0].attrs[
            'NetworkSettings']['Networks'].values())[0]['IPAddress']


def _connect_girder_client_to_local_dsa(ip):
    # connect a girder client to the local DSA docker
    apiUrl = 'http://%s:8080/api/v1' % ip
    gc = girder_client.GirderClient(apiUrl=apiUrl)
    gc.authenticate('admin', 'password')
    return gc


def _connect_to_existing_local_dsa():
    client = docker.from_env(version='auto')
    ipaddr = _get_htk_ipaddr(client)
    return _connect_girder_client_to_local_dsa(ipaddr)


def _create_and_connect_to_local_dsa():
    # create a local dsa docker and connect to it
    cwd = os.getcwd()
    thisDir = os.path.dirname(os.path.realpath(__file__))
    externdatadir = os.path.join(thisDir, '..', '.tox', 'externaldata')
    if not os.path.exists(externdatadir):
        os.makedirs(externdatadir)
    os.chdir(thisDir)
    outfilePath = os.path.join(
        tempfile.gettempdir(), 'histomicstk_test_girder_log.txt')
    with open(outfilePath, 'w') as outfile:

        # build a DSA docker container locally
        proc = subprocess.Popen([
            'docker-compose', 'up', '--build'],
            close_fds=True, stdout=outfile, stderr=outfile)

        os.chdir(cwd)
        timeout = time.time() + 1200

        # connect to docker and take a look at all its containers
        client = docker.from_env(version='auto')
        while time.time() < timeout:
            try:
                ipaddr = _get_htk_ipaddr(client)
                if ipaddr:
                    gc = _connect_girder_client_to_local_dsa(ipaddr)
                    break
            except Exception:  # noqa
                # warnings.warn(
                #     "Looks like the local DSA docker image is still "
                #     "initializing. Will wait a few seconds and try again.",
                # )
                time.sleep(0.1)

    return gc, proc


# TODO -- refactor to session scope by figuring out pytest issue (bug?)
# See https://docs.pytest.org/en/latest/fixture.html
# Note:
#     We could use scope='session' to create the DSA docker instance once
#     and reuse it for all test modules. However, pytest seems to have a
#     bug when yield is used (as oppoed to return) and it does not run the
#     teardown code properly. Instead, the girderClient fixture is called
#     again between modules, causing a stopIteration error.
#     Until this bug is fixed, we restrict the scope to the "module" level
#     to ensure: 1. Safe teardown, and 2. That edits to the database
#     done by one module do not carry over to the next module.
#

# @pytest.fixture(scope='session')
@pytest.fixture(scope='module')
def girderClient():
    """
    Yield an authenticated girder client that points to the server.

    If a local girder server docker is running, this will connect to it,
    otherwise, this will spin up a local girder server, load it with some
    initial data, and connect to it.

    NOTE: The default behavior initializes the docker image once per module and
    re-uses it for all tests. This means whatever one unit test changes in
    the DSA database is persistent for the next unit test. So if, for example,
    you remove one annotation as part of the first unit test, the next unit
    test will not have access to that annotation. Once all the unit tests are
    done, the database is torn down.

    If, instead, if you would like to run tests *repeatedly* (i.e. prototyping)
    , or you would like the changes written by tests in one module to be
    carried over to the next test module, you may prefer to start the server
    manually. That way you won't worry about unknown wait time till the local
    server is fully initialized. To manually start a DSA docker image, navigate
    to the directory where this file exists, and start the container:
    $ cd HistomicsTK/tests/
    $ docker-compose up --build

    Of course, you need to have docker installed and to either
    run this as sudo or be added to the docker group by the system admins.
    """
    try:
        # First we try to connect to any existing local DSA docker
        yield _connect_to_existing_local_dsa()

    except Exception as e:
        warnings.warn(
            e.__repr__() + "\n"
            "Looks like there's no existing local DSA docker running; "
            "will create one now and try again.",
        )
        # create a local dsa docker and connect to it
        gc, proc = _create_and_connect_to_local_dsa()

        yield gc
        proc.terminate()
        proc.wait()
