#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import json
import sys

from setuptools import find_packages

try:
    from skbuild import setup
except ImportError:
    print('scikit-build is required to build from source.', file=sys.stderr)
    print('Please run:', file=sys.stderr)
    print('', file=sys.stderr)
    print('  python -m pip install scikit-build')
    sys.exit(1)


# import numpy
# from Cython.Build import cythonize
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

# ext_compiler_args = ["-std=c++11", "-O2"]

# if sys.platform == "darwin":  # osx
#    ext_compiler_args.append("-mmacosx-version-min=10.9")

setup(
    name='histomicstk',
    version=pkginfo['version'],
    description=pkginfo['description'],
    long_description=readme + '\n\n' + history,
    author='Kitware, Inc.',
    author_email='developers@digitalslidearchive.net',
    url='https://github.com/DigitalSlideArchive/HistomicsTK',
    packages=find_packages(include=['histomicstk*']),
    package_dir={
        'histomicstk': 'histomicstk',
    },
    include_package_data=True,
    # package_data={
    #    '': ['*.rst', '*.txt', 'LICENSE*', '*.json', '*.xml', '*.pyx'],
    #     'histomicstk': ['*.rst', '*.txt', 'LICENSE*', '*.json', '*.xml', '*.pyx'],
    # },
    install_requires=requirements,
    license=license_str,
    keywords='histomicstk',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    # test_suite='plugin_tests',
    # tests_require=test_requirements,
    zip_safe=False,
)
