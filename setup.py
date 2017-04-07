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

# cython extensions
ext_compiler_args = ["-std=c++11", "-O2"]

if sys.platform == "darwin":  # osx
    ext_compiler_args.append("-mmacosx-version-min=10.9")

ext_list = [
    Extension(
        "histomicstk.segmentation.label.trace_boundaries_cython",
        sources=["histomicstk/segmentation/label/trace_boundaries_cython.pyx",
                 "histomicstk/segmentation/label/trace_boundaries_opt.cpp"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=ext_compiler_args,
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
