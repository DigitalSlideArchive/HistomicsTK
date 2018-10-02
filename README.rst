==============================================================
HistomicsTK |build-status| |codecov-io| |code-health| |gitter|
==============================================================

.. |build-status| image:: https://travis-ci.org/DigitalSlideArchive/HistomicsTK.svg?branch=master
    :target: https://travis-ci.org/DigitalSlideArchive/HistomicsTK
    :alt: Build Status

.. |codecov-io| image:: https://codecov.io/github/DigitalSlideArchive/HistomicsTK/coverage.svg?branch=master
    :target: https://codecov.io/github/DigitalSlideArchive/HistomicsTK?branch=master
    :alt: codecov.io

.. |code-health| image:: https://landscape.io/github/DigitalSlideArchive/HistomicsTK/master/landscape.svg?style=flat
   :target: https://landscape.io/github/DigitalSlideArchive/HistomicsTK/master
   :alt: Code Health

.. |gitter| image:: https://badges.gitter.im/DigitalSlideArchive/HistomicsTK.svg
   :alt: Join the chat at https://gitter.im/DigitalSlideArchive/HistomicsTK
   :target: https://gitter.im/DigitalSlideArchive/HistomicsTK?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

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

HistomicsTK can be used in two ways:

- **As a pure Python package**: This is intended to enable algorithm
  researchers to use and/or extend the analytics functionality within
  HistomicsTK in Python. HistomicsTK provides algorithms for fundamental
  image analysis tasks such as color normalization, color deconvolution,
  cell-nuclei segmentation, and feature extraction. Please see the
  `api-docs <https://digitalslidearchive.github.io/HistomicsTK/api-docs.html>`__
  and `examples <https://digitalslidearchive.github.io/HistomicsTK/examples.html>`__
  for more information.

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

