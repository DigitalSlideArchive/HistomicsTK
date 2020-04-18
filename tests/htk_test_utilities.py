import hashlib
import os
import pytest
import requests
import six
import subprocess
import tempfile
import time
import warnings
import docker
import girder_client


_checkedPaths = {}


def deleteIfWrongHash(destpath, hashvalue):
    """
    Check if a file at a path has a particular sha512 hash.  If not, delete it.
    If the file has been checked once, don't check it again.

    :param destpath: the file path.
    :param hashvalue: the sha512 hash hexdigest.
    """
    if os.path.exists(destpath) and destpath not in _checkedPaths and hashvalue:
        sha512 = hashlib.sha512()
        with open(destpath, 'rb') as f:
            while True:
                data = f.read(1024 * 1024)
                if not data:
                    break
                sha512.update(data)
        if sha512.hexdigest() != hashvalue:
            os.unlink(destpath)
        else:
            _checkedPaths[destpath] = True


def externaldata(
        hashpath=None, hashvalue=None, destdir='externaldata', destname=None,
        sources='https://data.kitware.com/api/v1/file/hashsum/sha512/{hashvalue}/download'):
    """
    Get a file from an external data source.  If the file has already been
    downloaded, check that it has the correct hash.

    :param hashpath: an optional path to a file that contains the hash value.
        There may be white space before or after the hashvalue.
    :param hashvalue: if no hashpath is specified, use this as a hash value.
    :param destdir: the location to store downloaded files.
    :param destname: if specified, the name of the file.  If hashpath is used
        and this is None, the basename of the hashpath is used for the
        destination name.
    :param sources: a string or list of strings that are url templates.
        `{hashvalue}` is replaced with the hashvalue.
    :returns: the path to the downloaded file.
    """
    if isinstance(sources, six.string_types):
        sources = [sources]
    curDir = os.path.dirname(os.path.realpath(__file__))
    if hashpath:
        hashvalue = open(os.path.join(curDir, hashpath)).read().strip()
        if destname is None:
            destname = os.path.splitext(os.path.basename(hashpath))[0]
    realdestdir = os.path.join(os.environ.get('TOX_WORK_DIR', curDir), destdir)
    destpath = os.path.join(realdestdir, destname)
    deleteIfWrongHash(destpath, hashvalue)
    if not os.path.exists(destpath):
        for source in sources:
            try:
                request = requests.get(source.format(hashvalue=hashvalue), stream=True)
                request.raise_for_status()
                if not os.path.exists(realdestdir):
                    os.makedirs(realdestdir)
                sha512 = hashlib.sha512()
                with open(destpath, 'wb') as out:
                    for buf in request.iter_content(65536):
                        out.write(buf)
                        sha512.update(buf)
                if os.path.getsize(destpath) == int(request.headers['content-length']):
                    if hashvalue and sha512.hexdigest() != hashvalue:
                        raise Exception('Download has wrong hash value - %s' % destpath)
                    break
                raise Exception('Incomplete download (got %d of %d) of %s' % (
                    os.path.getsize(destpath),
                    int(request.headers['content-length']), destpath))
            except Exception:
                pass
            if os.path.exists(destpath):
                os.unlink(destpath)
    if not os.path.exists(destpath):
        raise Exception('Failed to get external data %s' % destpath)
    return destpath


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
            except Exception as e:
                warnings.warn(e.__repr__(), RuntimeWarning)
                warnings.warn(
                    "Looks like the local DSA docker image is still "
                    "initializing. Will wait a few seconds and try again.",
                    RuntimeWarning
                )
                time.sleep(0.1)

    return gc, proc


@pytest.fixture(scope='session')
def girderClient():
    """
    Yield an authenticated girder client that points to the server.

    If a local girder server docker is running, this will connect to it,
    otherwise, this will spin up a local girder server, load it with some
    initial data, and connect to it.

    NOTE: If you are running tests multiple locally with pytest, you may prefer
    to start the local girder server manually (don't worry, it's
    just one command). That way, the unit tests are faster because the DSA docker
    container is only initialized once. This also has the added benefit of not
    having to worry about unknown wait time till the local server is fully
    initialized.To manually start a DSA docker image, navigate to the directory
    where this file exists, and start the container. like this ..
    $ cd HistomicsTK/tests/
    $ docker-compose up --build
    Of course, you need to have docker installed and to either
    run this as sudo or be added to the docker group by the system admins.
    """
    try:
        # First we try to connect to any existing local DSA docker
        yield _connect_to_existing_local_dsa()

    except Exception as e:
        warnings.warn(e.__repr__(), RuntimeWarning)
        warnings.warn(
            "Looks like there's no existing local DSA docker running; "
            "will create one now and try again.",
            RuntimeWarning
        )
        # create a local dsa docker and connect to it
        gc, proc = _create_and_connect_to_local_dsa()

        yield gc
        proc.terminate()
        proc.wait()
