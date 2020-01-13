================================================
HistomicsTK |build-status| |codecov-io| |gitter|
================================================

.. |build-status| image:: https://travis-ci.org/DigitalSlideArchive/HistomicsTK.svg?branch=master
    :target: https://travis-ci.org/DigitalSlideArchive/HistomicsTK
    :alt: Build Status

.. |codecov-io| image:: https://codecov.io/github/DigitalSlideArchive/HistomicsTK/coverage.svg?branch=master
    :target: https://codecov.io/github/DigitalSlideArchive/HistomicsTK?branch=master
    :alt: codecov.io

.. |gitter| image:: https://badges.gitter.im/DigitalSlideArchive/HistomicsTK.svg
   :target: https://gitter.im/DigitalSlideArchive/HistomicsTK?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
   :alt: Join the chat at https://gitter.im/DigitalSlideArchive/HistomicsTK

HistomicsTK is a Python and REST API for the analysis of Histopathology images
in association with clinical and genomic data. 

Histopathology, which involves the examination of thin-slices of diseased
tissue at a cellular resolution using a microscope, is regarded as the gold
standard in clinical diagnosis, staging, and prognosis of several diseases
including most types of cancer. The recent emergence and increased clinical
adoption of whole-slide imaging systems that capture large digital images of
an entire tissue section at a high magnification, has resulted in an explosion
of data. Compared to the related areas of radiology and genomics, there is a
dearth of mature open-source tools for the management, visualization and
quantitative analysis of the massive and rapidly growing collections of
data in the domain of digital pathology. This is precisely the gap that
we aim to fill with the development of HistomicsTK.

Developed in coordination with the `Digital Slide Archive`_ and
`large_image`_, HistomicsTK aims to serve the needs of both
pathologists/biologists interested in using state-of-the-art algorithms
to analyze their data, and algorithm researchers interested in developing
new/improved algorithms and disseminate them for wider use by the community.

You may view the following introductory videos for more information about
DSA and HistomicsTK:

- General overview: https://www.youtube.com/watch?v=NenUKZaT--k

- Simple annotation and data management tutorial: https://www.youtube.com/watch?v=HTvLMyKYyGs

HistomicsTK can be used in two ways:

- **As a pure Python package**: This is intended to enable algorithm
  researchers to use and/or extend the analytics functionality within
  HistomicsTK in Python. HistomicsTK provides algorithms for fundamental
  image analysis tasks such as color normalization, color deconvolution,
  cell-nuclei segmentation, and feature extraction. Please see the
  `api-docs <https://digitalslidearchive.github.io/HistomicsTK/api-docs.html>`__
  and `examples <https://digitalslidearchive.github.io/HistomicsTK/examples.html>`__
  for more information.
  
  Installation instructions on Linux:
  
  *To install HistomicsTK using PyPI*::
  
  $ python -m pip install histomicstk
  
  *To install HistomicsTK from source*::
  
  $ git clone https://github.com/DigitalSlideArchive/HistomicsTK/
  $ cd HistomicsTK/
  $ python -m pip install setuptools-scm Cython>=1.25.2 scikit-build>=0.8.1 cmake>=0.6.0 numpy>=1.12.1
  $ python -m pip install -e .

  HistomicsTK uses the `large_image`_ library to read and various microscopy
  image formats.  Depending on your exact system, installing the necessary 
  libraries to support these formats can be complex.  There are some
  non-official prebuilt libraries available for Linux that can be included as
  part of the installation by specifying 
  ``pip install histomicstk --find-links https://girder.github.io/large_image_wheels``.
  Note that if you previously installed HistomicsTK or large_image without
  these, you may need to add ``--force-reinstall --no-cache-dir`` to the
  ``pip install`` command to force it to use the find-links option.

  The system version of various libraries are used if the ``--find-links``
  option is not specified.  You will need to use your package manager to
  install appropriate libraries (on Ubuntu, for instance, you'll need 
  ``libopenslide-dev`` and ``libtiff-dev``).

- **As a server-side Girder plugin for web-based analysis**: This is intended
  to allow pathologists/biologists to apply analysis modules/pipelines
  containerized in HistomicsTK's docker plugins on data over the web. Girder_
  is a Python-based framework (under active development by Kitware_) for
  building web-applications that store, aggregate, and process scientific data.
  It is built on CherryPy_ and provides functionality for authentication,
  access control, customizable metadata association, easy upload/download of
  data, an abstraction layer that exposes data stored on multiple backends
  (e.g. Native file system, Amazon S3, MongoDB GridFS) through a uniform
  RESTful API, and most importantly an extensible plugin framework for
  building server-side analytics apps. To inherit all these capabilities,
  HistomicsTK is being developed to act also as a Girder plugin in addition
  to its use as a pure Python package. To further support web-based analysis,
  HistomicsTK depends on three other Girder plugins: (i) girder_worker_ for
  distributed task execution and monitoring, (ii) large_image_ for displaying,
  serving, and reading large multi-resolution images produced by whole-slide
  imaging systems, and (iii) slicer_cli_web_ to provide web-based RESTFul
  access to image analysis pipelines developed as `slicer execution model`_
  CLIs and containerized using Docker.

Please refer to https://digitalslidearchive.github.io/HistomicsTK/ for more information.

For questions, comments, or to get in touch with the maintainers, head to our
`Discourse forum`_, or use our `Gitter Chatroom`_.

This work is funded by the NIH grant U24-CA194362-01_.

.. _Digital Slide Archive: http://github.com/DigitalSlideArchive
.. _Docker: https://www.docker.com/
.. _Kitware: http://www.kitware.com/
.. _U24-CA194362-01: http://grantome.com/grant/NIH/U24-CA194362-01

.. _CherryPy: http://www.cherrypy.org/
.. _Girder: http://girder.readthedocs.io/en/latest/
.. _girder_worker: http://girder-worker.readthedocs.io/en/latest/
.. _large_image: https://github.com/girder/large_image
.. _slicer_cli_web: https://github.com/girder/slicer_cli_web
.. _slicer execution model: https://www.slicer.org/slicerWiki/index.php/Slicer3:Execution_Model_Documentation
.. _Discourse forum: https://discourse.girder.org/c/histomicstk
.. _Gitter Chatroom: https://gitter.im/DigitalSlideArchive/HistomicsTK?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

