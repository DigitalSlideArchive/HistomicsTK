==========================
HistomicsTK docker scripts
==========================

Description
===========

This folder contains a set of scripts that are convenient to develop
histomicsTK inside its docker container.

The following environment variable need to be defined for these scripts
to run:

* ``HISTOMICS_TESTDATA_FOLDER``: Folder in which the test data will be installed
  on the host computer. This allows to not download the test data every time,
  but instead keep it directly on the host computer. If the container is removed,
  there is no need to download the data again.

Scripts
=======

* ``deploy.sh``: wrapper script around ``deploy_docker.py`` HistomicsTK script to mount
  local host source and data folders. ``deploy_docker.py`` arguments can be added to the
  command line.
* ``configure.sh``: configure plugin inside docker container, and compile Cython files.
  This script needs to be run once after the container has been created with ``deploy.sh``,
  to populate the host folders that are mounted inside the docker environment, and when
  histomicsTK Cython files are modified.
* ``build.sh``: Build girder inside docker container to build HistomicsTK plugin.
* ``test.sh``: Run girder tests inside container. Ctest arguments can be added to the
  command line.
* ``connect.sh``: convenience script to log in the HistomicsTK docker container (wrapper
  around ``docker exec``).

Usage
=====

A typical use case of these scripts is when one develops HistomicsTK locally on their computer.
It is possible to run everything inside docker containers to simplify deployment. This is typically
done using ``deploy_docker.py`` in the ``ansible`` folder with the command::

  $ cd HistomicsTK/ansible
  $ python deploy_docker.py start

If you use this script, a copy of the source code is created inside the histomicsTK docker container.
This is not ideal to develop as one has to update this code to test their improvements. Instead, one
can now use ``deploy.sh`` in the ``devops`` folder. This will mount their local histomicsTK source
folder in their container::

  $ cd HistomicsTK
  $ export HISTOMICS_TESTDATA_FOLDER=~/data/histomicsTK
  $ mkdir -p $HISTOMICS_TESTDATA_FOLDER
  $ devops/deploy.sh start --build

To simplify future calls, one can set the environment variables directly in their ``.bashrc`` file.

Once the containers are up and running, call ``configure.sh`` to compile the Cython files using your
mounted source folder::

  $ devops/configure.sh

Make sure that HistomicsTK Girder plugin is up to date by recompiling it::

  $ devops/build.sh

And finally run the tests as if you were inside the build folder::

  $ devops/test.sh -V
