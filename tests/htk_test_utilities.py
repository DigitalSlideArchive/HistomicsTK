import hashlib
import os
import pytest
import requests
import six
import subprocess
import tempfile
import time


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
                    os.path.getsize(destpath), int(request.headers['content-length'], destpath)))
            except Exception:
                pass
            if os.path.exists(destpath):
                os.unlink(destpath)
    if not os.path.exists(destpath):
        raise Exception('Failed to get external data %s' % destpath)
    return destpath


@pytest.fixture(scope='session')
def girderClient():
    """
    Spin up a local girder server, load it with some initial data, and return
    an authenticated girder client that points to the server.
    """
    import docker
    import girder_client

    cwd = os.getcwd()
    thisDir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(thisDir)
    outfilePath = os.path.join(tempfile.gettempdir(), 'histomicstk_test_girder_log.txt')
    with open(outfilePath, 'w') as outfile:

        # DOES NOT WORK LOCALLY!!!
        # 1- does not work inside pycharm, has to be ipython alone
        # 2- When I try this approach of making a subprocess to run the docker
        #    it's REEEEALLY slow and there's no way for me to know when the docker
        #    has actually finished starting the local DSA server. It takes quite long
        #    and the heuristic currently used (if I understand it) is based on
        #    just waiting and having a timeout after a while!
        # What I found to work better is to just start the DSA server in a separate
        # terminal using docker compose-up --build and THEN to use the
        # rest of the pipeline here once the server is up, by looking for the
        # Histomics container and doind all sorts of stuff with it. Of course, make
        # sure to use the yml file that David M provides.

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
                # search docker containers for the newly created DSA docker
                # container and fetch its IP address
                ipaddr = list(client.containers.list(
                    filters={'label': 'HISTOMICSTK_GC_TEST'})[0].attrs[
                        'NetworkSettings']['Networks'].values())[0]['IPAddress']
                if ipaddr:
                    # Now connect a girder client to the local DSA docker
                    apiUrl = 'http://%s:8080/api/v1' % ipaddr
                    gc = girder_client.GirderClient(apiUrl=apiUrl)
                    gc.authenticate('admin', 'password')
                    break
            except Exception as e:
                print(e.__repr__())
                # Most likely the DSA docker it still initializing, so we wait
                # a bit and then try again
                time.sleep(0.1)
        yield gc
        proc.terminate()
        proc.wait()

