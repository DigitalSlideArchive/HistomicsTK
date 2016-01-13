#! /usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import json


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('plugin.json') as f:
    pkginfo = json.load(f)

with open('LICENSE') as f:
    license = f.read()

requirements = [
    # TODO: put package requirements here
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(name='histomicstk',
      version=pkginfo['version'],
      description=pkginfo['description'],
      long_description=readme + '\n\n' + history,
      author="Deepak Roy Chittajallu",
      author_email='cdeepakroy@gmail.com',
      url='https://github.com/DigitalSlideArchive/image_analysis',
      packages=['histomicstk'],
      package_dir={'histomicstk':
                   'histomicstk'},
      include_package_data=True,
      install_requires=requirements,
      license=license,
      zip_safe=False,
      keywords='histomicstk',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'License :: OSI Approved :: Apache Software License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2'
      ],
      test_suite='tests',
      tests_require=test_requirements)
