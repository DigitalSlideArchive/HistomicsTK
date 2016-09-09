.. highlight:: shell

============
Installation
============

HistomicsTK can be used in two ways:
1. **As a pure python toolkit**: This is intended to enable algorithm
   researchers to use and/or extend the analytics functionality within
   HistomicsTK in Python.
2. **As a server-side Girder_ plugin for web-based analysis**: This is intended
   to allow pathologists/biologists to apply analysis modules/pipelines in
   HistomicsTK's docker plugins on the data stored in the associated Girder_
   instance over the web.

Here, we describe how to install HistomicsTK for both these scenarios

Installing HistomicsTK as a Python toolkit
------------------------------------------

The following prerequisites should be installed before installing HistomicsTK:
- large_image_
- ctk_cli_ >= 1.3.1

Once the aforementioned pre-requisites are installed, HistomicsTK can be
installed as follows::

    $ git clone git@github.com:DigitalSlideArchive/HistomicsTK.git
    $ pip install -e HistomicsTK

Installing HistomicsTK as a Girder plugin using Vagrant and Ansible
-------------------------------------------------------------------

- Install virtual box - https://www.virtualbox.org/wiki/Downloads.
- Install vagrant - https://www.vagrantup.com/downloads.html
- $ `pip install ansible`
- $ `git clone git@github.com:DigitalSlideArchive/HistomicsTK.git`
- $ `cd HistomicsTK && vagrant up`

.. Girder_: http://girder.readthedocs.io/en/latest/
.. large_image_: https://github.com/DigitalSlideArchive/large_image
.. ctk_cli_: https://github.com/cdeepakroy/ctk-cli
