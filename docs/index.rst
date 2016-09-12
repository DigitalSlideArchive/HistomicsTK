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
data in the domain of digital pathology. The goal of HistomicsTK is to
full-fill this gap with funding from the NIH grant U24-CA194362-01_.

Developed in coordination with the `Digital Slide Archive`_ and
`large_image`_, HistomicsTK aims to serve the needs of both
pathologists/biologists interested in using state-of-the-art algorithms
to analyze their data, and algorithm researchers interested in developing
new/improved algorithms and disseminate them for wider use by the community.

HistomicsTK can be used in two ways:

1. **As a pure Python toolkit**: This is intended to enable algorithm
   researchers to use and/or extend the analytics functionality within
   HistomicsTK in Python. HistomicsTK provides algorithms for fundamental
   image analysis tasks such as color normalization, color deconvolution,
   nuclei segmentation, and feature extraction. Please see the
   :doc:`api-docs` and :doc:`examples` for more information.

2. **As a server-side Girder plugin for web-based analysis**: This is intended
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
   to its use as a pure Python toolkit.

Integrating your algorithms into HistomicsTK and the Digital Slide Archive
is made simple with Docker_ and the `Slicer Execution Model`_. This framework
gives developers the freedom to create portable algorithms and automatically
generate DSA UI elements for their algorithm, and exposes their work to a broad
community of users. Read more about this in the :doc:api-docs.

.. toctree::
   :maxdepth: 2

   installation
   usage
   api-docs
   examples
   contributing
   authors
   history

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _Digital Slide Archive: http://github.com/DigitalSlideArchive
.. _Docker: https://www.docker.com/
.. _large_image: https://github.com/DigitalSlideArchive/large_image
.. _Slicer Execution Model: https://www.slicer.org/slicerWiki/index.php/Slicer3:Execution_Model_Documentation
.. _U24-CA194362-01: http://grantome.com/grant/NIH/U24-CA194362-01
