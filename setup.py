#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
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


with open('README.rst', 'rt') as readme_file:
    readme = readme_file.read()


def prerelease_local_scheme(version):
    """
    Return local scheme version unless building on master in CircleCI.

    This function returns the local scheme version number
    (e.g. 0.0.0.dev<N>+g<HASH>) unless building on CircleCI for a
    pre-release in which case it ignores the hash and produces a
    PEP440 compliant pre-release version number (e.g. 0.0.0.dev<N>).
    """
    from setuptools_scm.version import get_local_node_and_date

    if os.getenv('CIRCLE_BRANCH') in {'master'}:
        return ''
    else:
        return get_local_node_and_date(version)


setup(
    name='histomicstk',
    use_scm_version={'local_scheme': prerelease_local_scheme},
    description='A Python toolkit for Histopathology Image Analysis',
    long_description=readme,
    long_description_content_type='text/x-rst',
    author='Kitware, Inc.',
    author_email='developers@digitalslidearchive.net',
    url='https://github.com/DigitalSlideArchive/HistomicsTK',
    packages=find_packages(include=['histomicstk*']),
    package_dir={
        'histomicstk': 'histomicstk',
    },
    include_package_data=True,
    setup_requires=[
        'setuptools-scm',
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
        'Pillow>=3.2.0',
        'pandas>=0.19.2',
        'scikit-image>=0.14.2',
        'scikit-learn>=0.18.1' + ('' if sys.version_info >= (3, ) else ',<0.21'),
        'imageio>=2.3.0',
        'shapely[vectorized]',
        'opencv-python',
        # deep learning packages
        'h5py>=2.7.1',
        'keras>=2.0.8',
        'tensorflow>=1.11',
        # dask packages
        'dask>=1.1.0',
        'distributed>=1.21.6',
        'tornado',
        'fsspec>=0.3.3;python_version>="3"',
        # large image sources
        'large-image-source-tiff',
        'large-image-source-openslide',
        'large-image-source-ometiff',
        'large-image-source-pil',
        # for interaction with girder
        'girder_client',
    ],
    license='Apache Software License 2.0',
    keywords='histomicstk',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    zip_safe=False,
)
