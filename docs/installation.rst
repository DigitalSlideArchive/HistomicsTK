.. highlight:: shell

============
Installation
============

HistomicsTK can be used in two ways:

1. **As a pure python toolkit**: This is intended to enable algorithm
   researchers to use and/or extend the analytics functionality within
   HistomicsTK in Python.

2. **As a server-side Girder plugin for web-based analysis**: This is intended
   to allow pathologists/biologists to apply analysis modules/pipelines
   containerized in HistomicsTK's docker plugins on data over the web. Girder_
   is a Python-based framework (under active development by Kitware_) for
   building web-applications that store, aggregate, and process scientific data.
   It is built on CherryPy_ and provides functionality for authentication,
   access control, customizable metadata association, easy upload/download of
   data, an abstraction layer that exposes data on multiple backends
   (e.g. Native file system, Amazon S3, MongoDB GridFS) through a uniform
   RESTful API, and most importantly an extensible plugin framework for
   server-side analytics. To inherit all these capabilities, HistomicsTK is
   being developed to act as a Girder plugin in addition to its use as a pure
   Python toolkit.


Here, we describe how to install HistomicsTK for both these scenarios

Installing HistomicsTK as a Python toolkit
------------------------------------------

HistomicsTK depends on large_image_ for reading large multi-resolution
whole-slide images in a tiled fashion. Please see the Github repo of
large_image to find out how to install it as a Python toolkit/package.

HistomicsTK also leverages the functionality of a number of scientific python
packages including numpy_, scipy_, scikit-image_, scikit-learn_,
and pandas_. We recommend using anaconda to ease the cross-platform
installation of these packages all of which are listed in
:doc:`requirments_c_conda.txt`.

Once large_image is installed, HistomicsTK can be installed as follows::

    $ git clone git@github.com:DigitalSlideArchive/HistomicsTK.git
    $ cd HistomicsTK
    $ conda install --yes libgfortran==1.0 setuptools==19.4 --file requirements_c_conda.txt
    $ python setup.py install

We are working on getting HistomicsTK onto PyPI so it can easily be pip
installed from there.

Installing HistomicsTK as a Girder plugin using Vagrant and Ansible
------------------------------------------------------------------------

- Download and install virtual box - https://www.virtualbox.org/wiki/Downloads
- Download and install vagrant - https://www.vagrantup.com/downloads.html
- ``pip install ansible``
- ``git clone git@github.com:DigitalSlideArchive/HistomicsTK.git``
- ``cd HistomicsTK && vagrant up``

The Girder instance can then be accessed at http://localhost:8009.

The front-end UI that allows you to apply analysis modules in HistomicsTK's
docker plugins on data stored in Girder can be accessed at
http://localhost:8009/histomicstk.

You can also ssh into the vagrant virtual box using the command ``vagrant ssh``.
HistomicsTK and its dependencies are installed ad ``/opt/histomicstk``.

.. _CherryPy: http://www.cherrypy.org/
.. _ctk_cli: https://github.com/cdeepakroy/ctk-cli
.. _Girder: http://girder.readthedocs.io/en/latest/
.. _Kitware: http://www.kitware.com/
.. _large_image: https://github.com/DigitalSlideArchive/large_image
.. _numpy: http://www.numpy.org/
.. _pandas: http://pandas.pydata.org/
.. _scikit-image: http://scikit-image.org/
.. _scikit-learn: http://scikit-learn.org/stable/
.. _scipy: https://www.scipy.org/


