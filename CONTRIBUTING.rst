.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/DigitalSlideArchive/HistomicsTK/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug"
is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "feature"
is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

HistomicsTK could always use more documentation, whether as part of the
official HistomicsTK docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/DigitalSlideArchive/HistomicsTK/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `HistomicsTK` for local development.

1. Fork the `HistomicsTK` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/HistomicsTK.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development::

    $ mkvirtualenv HistomicsTK
    $ cd HistomicsTK/
    $ python -m pip install setuptools-scm Cython>=0.25.2 scikit-build>=0.8.1 cmake>=0.6.0 numpy>=1.12.1
    $ python setup.py develop
    
Of course, any type of virtual python environment will do. These instructions are equally applicable inside `conda` environments.

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests (see notes below).
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for Python 2.6, 2.7, and for PyPy. Check
   https://travis-ci.org/DigitalSlideArchive/HistomicsTK/pull_requests
   and make sure that the tests pass for all supported Python versions.

Unit Testing Notes
----------------------------

HistomicsTK can be used in two ways: as a pure python package and as a server-side
plugin, there are two 'modes' of testing. 

* Ordinary unit testing:

  If your newly added method/function does not need to run on the server side
  feel free to use python's ``unittest`` module to create unit tests that
  ensure that your method works when used as a stand-alone python package.
  Use the standard ``your_python_file_test.py`` naming convention.

* Server side testing:

  If your newly added method/function uses girder and is meant to be run
  on the server side, you will need to have unit tests that run on the
  server side. To find examples for these, go to ``./plugin_tests/``.
  Specifically, ``example_test.py`` provides a schema that you can use.
  If your tests require access to girder items (slide, JSON, etc), it would be
  ideal to refactor how the tests are done so that they download files from
  ``data.kitware.com``, then start a test instance of Girder, upload the files
  to it, and then proceed with the girder_client calls.  This has the virtue
  that we would not need to have the credentials for an external girder instance.
  If you run the tests multiple times, it will only download the test files once.
  For example, ckeck out ``./plugin_tests/annotations_to_masks_handler.py``,
  and notice how ``GirderClientTestCase`` is used to provide access to the
  slide and annotations, which are referenced using ``.sha512`` hash that
  is present in ``./plugin_tests/data/``. Please contact the owners if you
  have questions about this or need support on how to host your test data
  on ``data.kitware.com`` to make this work.


Travis Integration Notes
----------------------------

When you submit a pull request to merge your branch with master, it will be
automatically submitted to Travis CI for continuous integration. In plain
English, your new branch will be tested to make sure it complies with the
standardized coding and documentation style of this repo. If you'd like
to help the organizers integrate your changes seamlessly, check to see
if the travis CI was passed. Otherwise, examine for errors and see if you
can get them fixed. Oftentimes, the errors originate from code and docstring
formatting and/or integration of jupyter notebooks into the documentaion
examples. Here are some pointers to help you handle some of these issues:

* Consider using ``flake8`` package to check if you comply with the
  formatting standard. HistomicsTK uses PEP8 standard with some options
  turned off. The ``flake8`` parameters we use can be found in:
  https://github.com/girder/girder/blob/2.x-maintenance/setup.cfg

  For example::

  $ flake8 your_python_file.py

  You can find ``flake8`` at: http://flake8.pycqa.org/en/latest/

  If you like using Vim editor, there is a tool to integrate ``flake8``
  with Vim for easy correction of errors at: https://github.com/nvie/vim-flake8

* If your text editor does not already have this feature, consider using the
  package ``autopep8`` to comply with PEP8 standard: https://github.com/hhatto/autopep8 .
  for example::

  $ autopep8 --in-place --aggressive your_python_file.py

* Consider using ``pydocstyle`` to check if you comply with the PEP257
  standard for docstrings: https://github.com/PyCQA/pydocstyle . For example::

  $ pydocstyle your_python_file.py

* If your text editor does not already do this, consider using ``docformatter``
  to fix docstrings to standard: https://pypi.org/project/docformatter/ . For
  example::

  $ docformatter --in-place --pre-summary-newline --blank your_python_file.py

* If you added new functionality, consider adding the documentation under
  ``doc`` in the form of rst files. Also consider creating Jupyter
  Notebooks to showcase functionality under ``doc/examples/``. The documentation
  is automatically generated using ``sphinx`` when you push your pull request and
  it gets submitted for travis integration. If you added documentation, consider
  checking if ``sphinx`` throws errors offline. you may install it from:
  https://www.sphinx-doc.org/en/master/index.html
  create a folder for the generated documentation to be saved, let's say
  ``~/HistomicsTK_test_build/`` . Then you may run something like::

  $ cd HistomicsTK
  $ sphinx-build ./docs/ ~/HistomicsTK_test_build/ 2>&1 | tee out.log

  Then you may check the file ``out.log`` for build errors.
