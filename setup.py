#! /usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import json
from pkg_resources import parse_requirements


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('plugin.json') as f:
    pkginfo = json.load(f)

with open('LICENSE') as f:
    license = f.read()

try:
    with open('requirements.txt') as f:
        ireqs = parse_requirements(f.read())
except Exception:
    pass
requirements = [str(req) for req in ireqs]

test_requirements = [
    # TODO: put package test requirements here
]

setup(name='histomicstk',
      version=pkginfo['version'],
      description=pkginfo['description'],
      long_description=readme + '\n\n' + history,
      author='Kitware, Inc.',
      author_email='developers@digitalslidearchive.net',
      url='https://github.com/DigitalSlideArchive/HistomicsTK',
      packages=['histomicstk'],
      package_dir={'histomicstk':
                   'histomicstk'},
      include_package_data=True,
      install_requires=requirements,
      license=license,
      zip_safe=False,
      keywords='histomicstk',
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Environment :: Console',
          'License :: OSI Approved :: Apache Software License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Software Development :: Libraries :: Python Modules',
      ],
      test_suite='tests',
      tests_require=test_requirements)
