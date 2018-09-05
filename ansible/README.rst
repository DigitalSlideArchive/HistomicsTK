===================
Install HistomicsTK
===================

There are several ways to install HistomicsTK and the Digital Slide Archive.  If you intend to use the interface, use the Docker installation.  If you need to develop the source code, the Vagrant installation is the easiest method.  If you are using Ubuntu 16.04, you can install HistomicsTK on your local system.

.. __methods

Installing via Docker
---------------------

This method should work on any system running Docker.  It has been tested with a variety of Ubuntu and CentOS distributions.

Prerequisites
#############

Install git, python-pip, and docker.io.  On Ubuntu, this can be done via::

    sudo apt-get update
    sudo apt-get install git docker.io python-pip

The current user needs to be a member of the docker group::

    sudo usermod -aG docker `id -u -n`

After which, you will need to log out and log back in.

Install the python docker module::

    sudo pip install docker

Get the HistomicsTK repository::

    git clone https://github.com/DigitalSlideArchive/HistomicsTK

Deploy
######

::

    cd HistomicsTK/ansible
    python deploy_docker.py start

There are many options that can be used along with the ``deploy_docker.py`` command, use ``deploy_docker.py --help`` to list them.

By default, the deployment places all database, log, and assetstore files in the ``~/.histomicstk`` directory.  HistomicsTK is run on localhost at port 8080.

Update an installation
######################

::

    cd HistomicsTK/ansible
    # Make sure you have the latest version of the deploy_docker script
    git pull
    # Make sure you have the latest docker images.  This uses the pinned
    # versions of the docker images -- add --latest to use the latest built
    # images.
    python deploy_docker.py pull
    # stop and remove the running docker containers for HistomicsTK
    python deploy_docker.py rm
    # Restart and provision the new docker containers.  Use the same
    # command-line parameters as you originally used to start HistomicsTK the
    # first time.
    python deploy_docker.py start

Installing via Vagrant
----------------------

This method can work on Linux, Macintosh, or Windows.

Prerequisites
#############

Install VirtualBox, Vagrant, and git:

- Download and install git - https://git-scm.com/downloads
- Download and install virtual box - https://www.virtualbox.org/wiki/Downloads
- Download and install vagrant - https://www.vagrantup.com/downloads.html

Get the HistomicsTK repository::

    git clone https://github.com/DigitalSlideArchive/HistomicsTK

Deploy
######

::

    cd HistomicsTK
    vagrant up

The Girder instance can then be accessed at http://localhost:8009. Any image
placed in the sample_images subdirectory of the directory where HistomicsTK
is cloned directory will be seen in the TCGA collection of Girder.

The front-end UI that allows you to apply analysis modules in HistomicsTK's
docker plugins on data stored in Girder can be accessed at
http://localhost:8009/histomicstk.

You can also ssh into the vagrant virtual box using the command ``vagrant ssh``.
HistomicsTK and its dependencies are installed at the location
``/opt/histomicstk``.

Run tests
#########

Log in to the vagrant box::

    vagrant ssh

Inside the vagrant box, tests can be run by typing::

    cd /opt/histomicstk/build
    ctest -VV

Local installation on Ubuntu 16.04
----------------------------------

Due to the library dependencies of OpenJPEG, libtiff, OpenSlide, and vips, local installation may be hard to get fully working.  The local deployment scripts assume a reasonably plain instance of Ubuntu 16.04.

Prerequisites
#############

::

    sudo apt-get update
    sudo apt-get install -y --force-yes libssl-dev git python2.7-dev python-pip
    sudo pip install -U pip
    sudo pip install -U 'ansible<2.5'
    git clone https://github.com/DigitalSlideArchive/HistomicsTK

Deploy
######

::

    cd HistomicsTK/ansible
    ./deploy_local.sh

Note that if there are network issues, this deployment script does not automatically retry installation.  It may be necessary to delete partial files and run it again.
