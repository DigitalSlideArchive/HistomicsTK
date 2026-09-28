.. highlight:: shell

============
Installation
============

HistomicsTK can be used both as a pure Python toolkit for algorithm development
and as server-side Girder_ plugin for web-based analysis. Here, we describe how
to install HistomicsTK for both these scenarios.

Installing HistomicsTK as a Python toolkit
------------------------------------------

On Linux, HistomicsTK can be installed via pip.  You can specify the
``--find-links`` option to get prebuilt libraries for reading some common image
formats.  The installation command is::

    $ pip install histomicstk --find-links https://girder.github.io/large_image_wheels

For non-Linux systems, or to use system libraries for reading image formats,
please see the Github repo of large_image to find out how to install it as a
Python toolkit/package.  Once large_image is installed as a python package,
HistomicsTK can be installed as follows::

    $ git clone https://github.com/DigitalSlideArchive/HistomicsTK.git
    $ cd HistomicsTK
    $ pip install -e .

If you want the keras/tensorflow based functions of HistomicsTK to take advantage of the GPU,
then you will have to install the GPU version of tensorflow (tensorflow-gpu) after
installing HistomicsTK. See tensorflow installation instructions `here <https://www.tensorflow.org/install/>`__.

Installing HistomicsTK as a server-side Girder plugin
-----------------------------------------------------

When HistomicsTK is used as a server-side Girder_ plugin for web-based
analysis, the following three Girder plugins need to be installed:

- girder_worker_: A Girder plugin for distributed task execution.
- large_image_: A Girder plugin to create/serve/display large
  multi-resolution images produced by whole-slide imaging systems and a
  stand-alone Python package to read/write these images.
- slicer_cli_web_: A Girder plugin for providing web-based RESTFul access
  to image analysis pipelines developed as slicer execution model
  CLIs and containerized using Docker.

There are several methods that can be used to install HistomicsTK.  Each of these results in a fully deployed system.  Docker is often the easiest deployment.  Vagrant is the easiest development environment.

Installing Slicer CLI Docker Images for Analysis
------------------------------------------------

One of the primary uses of HistomicsTK as a Girder plugin is to analyze images.
Analyses are configured using Docker images conforming to the Slicer execution
model.  Different analysis Docker images can be installed by a system
administrator using the Web API (the link is on the bottom of the Girder web
page).

Selecting the Web API will show a list of API endpoints for Girder.  In the
``PUT`` ``/HistomicsTK/HistomicsTK/docker_image`` endpoint, you can add a list
of Docker images, such as ``["dsarchive/histomicstk:latest"]``, and then click
"Try it out".  This will pull the specified Docker image and make it available
to the HistomicsTK interface.  You could also use a specific tag (version) from
Docker instead of the ``latest`` tag.  Multiple versions and multiple images
can be installed so that they can be compared against each other.

For an example of how to make a Docker image with a Slicer CLI-compatible
interface, see `here <https://github.com/cdeepakroy/slicer_cli_web_plugin>`__.

.. _Girder: http://girder.readthedocs.io/en/latest/
.. _girder_worker: http://girder-worker.readthedocs.io/en/latest/
.. _Kitware: http://www.kitware.com/
.. _large_image: https://github.com/girder/large_image
.. _numpy: http://www.numpy.org/
.. _pandas: http://pandas.pydata.org/
.. _scikit-image: http://scikit-image.org/
.. _scikit-learn: http://scikit-learn.org/stable/
.. _scipy: https://www.scipy.org/
.. _slicer_cli_web: https://github.com/girder/slicer_cli_web
.. _anaconda: https://www.continuum.io/downloads
