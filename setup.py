#! /usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup, find_packages, Extension
except ImportError:
    from distutils.core import setup, find_packages
    from distutils.extension import Extension

import os
import json
import numpy
import sys
from Cython.Build import cythonize
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

test_requirements = [
    # TODO: Should we list Girder here?
]

ext_compiler_args = ["-std=c++11", "-O2"]

if sys.platform == "darwin":  # osx
    ext_compiler_args.append("-mmacosx-version-min=10.9")

ext_list = [
    Extension(
        "histomicstk.segmentation.label._trace_object_boundaries_cython",
        sources=[
            "histomicstk/segmentation/label/_trace_object_boundaries_cython.pyx",
        ],
        include_dirs=[numpy.get_include()],
        language="c++",
    ),

    Extension(
        "histomicstk.segmentation.nuclear._max_clustering_cython",
        sources=["histomicstk/segmentation/nuclear/_max_clustering_cython.pyx"],
        include_dirs=[numpy.get_include()]
    )
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
      tests_require=test_requirements,
      ext_modules = cythonize(ext_list)
)
