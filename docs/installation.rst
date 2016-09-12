.. highlight:: shell

============
Installation
============

As mentioned in the :doc:`index`, HistomicsTK can be used both as a pure
Python toolkit for algorithm development and as server-side Girder_ plugin
for web-based analysis. Here, we describe how to install HistomicsTK for both
these scenarios.

Installing HistomicsTK as a Python toolkit
------------------------------------------

HistomicsTK depends on large_image_ for reading large multi-resolution
whole-slide images in a tiled fashion. Please see the Github repo of
large_image to find out how to install it as a Python toolkit/package.

HistomicsTK also leverages the functionality of a number of scientific python
packages including numpy_, scipy_, scikit-image_, scikit-learn_,
and pandas_. We recommend using anaconda to ease the cross-platform
installation of these packages all of which are listed in
`requirments_c_conda.txt <https://github.com/DigitalSlideArchive/HistomicsTK/blob/master/requirements_c_conda.txt>`__.

Once large_image is installed, HistomicsTK can be installed as follows::

    $ git clone git@github.com:DigitalSlideArchive/HistomicsTK.git
    $ cd HistomicsTK
    $ conda install --yes libgfortran==1.0 setuptools==19.4 --file requirements_c_conda.txt
    $ python setup.py install

We are working on releasing HistomicsTK on PyPI so it can easily be pip
installed from there.

Installing HistomicsTK as a server-side Girder plugin using Vagrant
-------------------------------------------------------------------

To use HistomicsTK as a server-side Girder_ plugin for web-based analysis,
a few other Girder plugins need to be installed:

- girder_worker_: A distributed task execution engine
- large_image_: A Girder plugin to create/serve/display large
  multi-resolution images produced by whole-slide imaging systems and a
  stand-alone Python package to read/write these images.
- slicer_cli_web_: A Girder plugin for exposing slicer execution model CLIs
  over the web using Docker for containerization and girder_worker for
  distributed execution

We used Vagrant and Ansible to ease the installation of these plugins in
addition to HistomicsTK as follows:

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
HistomicsTK and its dependencies are installed at the location
``/opt/histomicstk``.

.. _CherryPy: http://www.cherrypy.org/
.. _ctk_cli: https://github.com/cdeepakroy/ctk-cli
.. _Girder: http://girder.readthedocs.io/en/latest/
.. _girder_worker:
.. _Kitware: http://www.kitware.com/
.. _large_image: https://github.com/DigitalSlideArchive/large_image
.. _numpy: http://www.numpy.org/
.. _pandas: http://pandas.pydata.org/
.. _scikit-image: http://scikit-image.org/
.. _scikit-learn: http://scikit-learn.org/stable/
.. _scipy: https://www.scipy.org/


