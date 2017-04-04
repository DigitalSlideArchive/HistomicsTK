.. highlight:: shell

============
Installation
============

As mentioned in the :doc:`index <index>`, HistomicsTK can be used both as a pure
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
and pandas_.  If you have not built libtiff or openjpeg locally, we recommend
installing anaconda_ to ease the cross-platform installation of these packages
all of which are listed in
`requirments_c_conda.txt <https://github.com/DigitalSlideArchive/HistomicsTK/blob/master/requirements_c_conda.txt>`__.
If libtiff or openjpeg were built locally, use pip to install these packages,
as otherwise the binaries may not match properly.

Once large_image is installed as a python package, HistomicsTK can be
installed as follows::

    $ git clone https://github.com/DigitalSlideArchive/HistomicsTK.git
    $ cd HistomicsTK
    $ conda config --add channels https://conda.binstar.org/cdeepakroy
    $ conda install --yes ctk-cli==1.4.1
    $ pip install --no-cache-dir -r requirements_c_conda.txt
    $ pip install --no-cache-dir -r requirements.txt -r requirements_c.txt
    $ python setup.py build_ext --inplace
    $ pip install .

If you build libtiff or openjpeg locally, ensure that the appropriate pip files
have also been installed locally so that they use the correct libraries::

    $ pip install --upgrade --force-reinstall --ignore-installed --no-cache-dir openslide-python Pillow
    $ pip install --upgrade --force-reinstall --ignore-installed 'git+https://github.com/pearu/pylibtiff@848785a6a9a4e2c6eb6f56ca9f7e8f6b32e523d5'

We are working on releasing HistomicsTK on PyPI so it can easily be pip
installed from there.

Installing HistomicsTK as a server-side Girder plugin using Vagrant
-------------------------------------------------------------------

When HistomicsTK is used as a server-side Girder_ plugin for web-based
analysis, the following three Girder plugins need to be installed:

- girder_worker_: A Girder plugin for distributed task execution.
- large_image_: A Girder plugin to create/serve/display large
  multi-resolution images produced by whole-slide imaging systems and a
  stand-alone Python package to read/write these images.
- slicer_cli_web_: A Girder plugin for providing web-based RESTFul access
  to image analysis pipelines developed as slicer execution model
  CLIs and containerized using Docker.

We used Vagrant and Ansible to ease the installation of these plugins in
addition to HistomicsTK as follows:

- Download and install virtual box - https://www.virtualbox.org/wiki/Downloads
- Download and install vagrant - https://www.vagrantup.com/downloads.html
- ``pip install ansible``
- ``git clone https://github.com/DigitalSlideArchive/HistomicsTK.git``
- ``cd HistomicsTK && vagrant up``

The Girder instance can then be accessed at http://localhost:8009. Any image
placed in the sample_images subdirectory of the directory where HistomicsTK
is cloned directory will be seen in the TCGA collection of Girder.

The front-end UI that allows you to apply analysis modules in HistomicsTK's
docker plugins on data stored in Girder can be accessed at
http://localhost:8009/histomicstk.

You can also ssh into the vagrant virtual box using the command ``vagrant ssh``.
HistomicsTK and its dependencies are installed at the location
``/opt/histomicstk``.

.. _Girder: http://girder.readthedocs.io/en/latest/
.. _girder_worker: http://girder-worker.readthedocs.io/en/latest/
.. _Kitware: http://www.kitware.com/
.. _large_image: https://github.com/DigitalSlideArchive/large_image
.. _numpy: http://www.numpy.org/
.. _pandas: http://pandas.pydata.org/
.. _scikit-image: http://scikit-image.org/
.. _scikit-learn: http://scikit-learn.org/stable/
.. _scipy: https://www.scipy.org/
.. _slicer_cli_web: https://github.com/girder/slicer_cli_web
.. _anaconda: https://www.continuum.io/downloads


