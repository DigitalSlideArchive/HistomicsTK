HistomicsTK
============

HistomicsTK is a Python and REST API for the analysis of Histopathology images. Developed in coordination with the `Digital Slide Archive`_, HistomicTK is intended for consumers of histopathology workflows and algorithms developers alike. It provides algorithms for fundamental image analysis tasks, algorithm pipelines for common histopathology workflows, and a framework for the easy distribution and integration of algorithms and pipelines.

To see examples of HistomicsTK's capabilities, check out the algorithm and pipeline examples (link) or visit the algorithm library (link) to learn about the filtering, normalization, segmentation and feature extraction capabilities.

Installing HistomicsTK is easy with pip. See the :doc:installation page for more details.

Integrating your algorithms into HistomicsTK and the Digital Slide Archive is made simple with Docker_ and the `Slicer Execution Model`_. This framework gives developers the freedom to create portable algorithms and automatically generate DSA UI elements for their algorithm, and exposes their work to a broad community of users. Read more about this in the :doc:api-docs.


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
.. _Slicer Execution Model: https://www.slicer.org/slicerWiki/index.php/Slicer3:Execution_Model_Documentation
