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

`HistomicsTK`_ is a Python package for the analysis of digital pathology images. It can function as a stand-alone library, or as a Digital Slide Archive plugin that allows users to invoke image analysis jobs through HistomicsUI. The functionality offered by HistomicsTK can be extended using `slicer cli web <https://github.com/girder/slicer_cli_web>`__ which allows developers to integrate their image analysis algorithms into DSA for dissemination through HistomicsUI. 

Whole-slide imaging captures the histologic details of tissues in large multiresolution images. Improvements in imaging technology, decreases in storage costs, and regulatory approval of digital pathology for primary diagnosis have resulted in an explosion of whole-slide imaging data. Digitization enables the application of computational image analysis and machine learning algorithms to characterize the contents of these images, and to understand the relationships between histology, clinical outcomes, and molecular data from genomic platforms. Compared to the related areas of radiology and genomics, open-source tools for the management, visualization, and analysis of digital pathology has lagged. To address this we have developed HistomicsTK in coordination with the `Digital Slide Archive`_ (DSA), a platform for managing and sharing digital pathology images in a centralized web-accessible server, and `HistomicsUI`_, a specialized user interface for annotation and markup of whole-slide images and for running image analysis tools and for scalable visualizing of dense outputs from image analysis algorithms. HistomicsTK aims to serve the needs of both pathologists/biologists interested in using state-of-the-art algorithms to analyze their data, and algorithm researchers interested in developing new/improved algorithms and disseminate them for wider use by the community.

HistomicsTK can be used in two ways:

- **As a pure Python package**: enables application of image analysis algorithms to data independent of the `Digital Slide Archive`_ (DSA). HistomicsTK provides a collection of fundamental algorithms for tasks such as color normalization, color deconvolution, nuclei segmentation, and feature extraction. Read more about these capabilities here:  `api-docs <https://digitalslidearchive.github.io/HistomicsTK/api-docs.html>`__ and `examples <https://digitalslidearchive.github.io/HistomicsTK/examples.html>`__ for more information.
  
  Installation instructions on Linux:
  
  *To install HistomicsTK using PyPI*::
  
  $ python -m pip install histomicstk
  
  *To install HistomicsTK from source*::
  
  $ git clone https://github.com/DigitalSlideArchive/HistomicsTK/
  $ cd HistomicsTK/
  $ python -m pip install setuptools-scm Cython>=1.25.2 scikit-build>=0.8.1 cmake>=0.6.0 numpy>=1.12.1
  $ python -m pip install -e .

  HistomicsTK uses the `large_image`_ library to read content from whole-slide and microscopy image formats. Depending on your exact system, installing the necessary libraries to support these formats can be complex.  There are some non-official prebuilt libraries available for Linux that can be included as part of the installation by specifying ``pip install histomicstk --find-links https://girder.github.io/large_image_wheels``. Note that if you previously installed HistomicsTK or large_image without these, you may need to add ``--force-reinstall --no-cache-dir`` to the ``pip install`` command to force it to use the find-links option.

  The system version of various libraries are used if the ``--find-links`` option is not specified.  You will need to use your package manager to install appropriate libraries (on Ubuntu, for instance, you'll need ``libopenslide-dev`` and ``libtiff-dev``).

- **As a image-processing task library for HistomicsUI and the Digital Slide Archive**: This allows end users to apply containerized analysis modules/pipelines over the web. See the `Digital Slide Archive`_ for installation instructions.

Refer to `our website`_ for more information.

For questions, comments, or to get in touch with the maintainers, head to our
`Discourse forum`_, or use our `Gitter Chatroom`_.


Previous Versions
-----------------

The HistomicsTK repository used to contain almost all of the Digital Slide Archive and HistomicsUI, and now container primarily code for image analysis algorithms and processing of annotation data.  The deployment and installation code and instructions for DSA have moved to the `Digital Slide Archive`_ repository.  The user interface and annotation functionality has moved to the `HistomicsUI`_ repository.

The deployment and UI code will eventually be removed from the master branch of this repository; any new development on those topics should be done in those locations.

Funding
-------

This work is funded by the NIH grant U24-CA194362-01_.

See Also
---------

**DSA/HistomicsTK project website:**
`Demos <https://digitalslidearchive.github.io/digital_slide_archive/demos-examples/>`_ |
`Success stories <https://digitalslidearchive.github.io/digital_slide_archive/success-stories/>`_

**Source repositories:** `Digital Slide Archive`_ | `HistomicsUI`_ | `large_image`_ | `slicer_cli_web`_

**Discussion:** `Discourse forum`_ | `Gitter Chatroom`_

.. Links for everythign above (not rendered):
.. _HistomicsTK: https://digitalslidearchive.github.io/digital_slide_archive/
.. _Digital Slide Archive: http://github.com/DigitalSlideArchive/digital_slide_archive
.. _HistomicsUI: http://github.com/DigitalSlideArchive/HistomicsUI
.. _large_image: https://github.com/girder/large_image
.. _our website: https://digitalslidearchive.github.io/digital_slide_archive/
.. _slicer execution model: https://www.slicer.org/slicerWiki/index.php/Slicer3:Execution_Model_Documentation
.. _slicer_cli_web: https://github.com/girder/slicer_cli_web
.. _Docker: https://www.docker.com/
.. _Kitware: http://www.kitware.com/
.. _U24-CA194362-01: http://grantome.com/grant/NIH/U24-CA194362-01
.. _Discourse forum: https://discourse.girder.org/c/histomicstk
.. _Gitter Chatroom: https://gitter.im/DigitalSlideArchive/HistomicsTK?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

