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


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('plugin.json') as f:
    pkginfo = json.load(f)

with open('LICENSE') as f:
    license_str = f.read()

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
    setup_requires=[
        'Cython>=0.25.2',
        'scikit-build>=0.8.1',
        'cmake>=0.6.0',
    ],
    install_requires=[
        'ctk-cli>=1.5',
        # scientific packages
        'nimfa>=1.3.2',
        'numpy>=1.12.1',
        'scipy>=0.19.0',
        'pandas>=0.19.2',
        'scikit-image>=0.14.2',
        'scikit-learn>=0.18.1',
        # deep learning packages
        'h5py>=2.7.1',
        'keras>=2.0.8',
        'tensorflow>=1.11',
        # dask packages
        'dask>=1.1.0',
        'distributed>=1.21.6',
        'tornado',
        # large image sources
        'large-image-source-tiff',
        'large-image-source-openslide',
        'large-image-source-pil',
    ],
    license=license_str,
    keywords='histomicstk',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    zip_safe=False,
)
