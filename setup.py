#! /usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

import os
import json
from pkg_resources import parse_requirements, RequirementParseError

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('plugin.json') as f:
    pkginfo = json.load(f)

with open('LICENSE') as f:
    license_str = f.read()

try:
    with open('requirements.txt') as f:
        ireqs = parse_requirements(f.read())
except RequirementParseError:
    raise
requirements = [str(req) for req in ireqs]

# if not on ReadTheDocs then add requirements depending on C libraries
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if not on_rtd:

    requirements_c_files = ['requirements_c_conda.txt',
                            'requirements_c.txt']

    for reqfile in requirements_c_files:
        try:
            with open(reqfile) as f:
                ireqs_c = parse_requirements(f.read())
        except RequirementParseError:
            raise
        cur_requirements = [str(req) for req in ireqs_c]
        requirements += cur_requirements

test_requirements = [
    # TODO: Should we list Girder here?
]

setup(name='histomicstk',
      version=pkginfo['version'],
      description=pkginfo['description'],
      long_description=readme + '\n\n' + history,
      author='Kitware, Inc.',
      author_email='developers@digitalslidearchive.net',
      url='https://github.com/DigitalSlideArchive/HistomicsTK',
      packages=find_packages(exclude=['doc']),
      package_dir={'histomicstk':
                   'histomicstk'},
      include_package_data=True,
      install_requires=requirements,
      license=license_str,
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
      test_suite='plugin_tests',
      tests_require=test_requirements)
