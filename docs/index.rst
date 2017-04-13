HistomicsTK
============

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
  :doc:`api-docs <api-docs>` and :doc:`examples <examples>` for more
  information.

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
  HistomicsTK depends on three other Girder plugins:

  - girder_worker_: A Girder plugin for distributed task execution.
  - large_image_: A Girder plugin to create/serve/display large
    multi-resolution images produced by whole-slide imaging systems and a
    stand-alone Python package for reading these images.
  - slicer_cli_web_: A Girder plugin for providing web-based RESTFul access
    to image analysis pipelines developed as `slicer execution model`_
    CLIs and containerized using Docker.

This work is funded by the National Cancer Institute (NCI) grant
U24-CA194362-01_.

.. toctree::
   :maxdepth: 2

   installation
   api-docs
   examples
   contributing
   authors

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

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
