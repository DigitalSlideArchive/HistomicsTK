#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

from setuptools import find_packages

try:
    from skbuild import setup
except ImportError:
    sys.stderr.write("""scikit-build is required to build from source or run tox.
Please run:
  python -m pip install scikit-build
""")
    # from setuptools import setup
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
    packages=find_packages(exclude=['tests', '*_test']),
    package_dir={
        'histomicstk': 'histomicstk',
    },
    include_package_data=True,
    install_requires=[
        # scientific packages
        'nimfa>=1.3.2',
        'numpy>=1.12.1',
        'scipy>=0.19.0',
        'Pillow>=3.2.0',
        'pandas>=0.19.2',
        'scikit-image>=0.14.2',
        'scikit-learn>=0.18.1' + ('' if sys.version_info >= (3, ) else ',<0.21'),
        'imageio>=2.3.0' + ('' if sys.version_info >= (3, ) else ',<2.8'),
        'shapely[vectorized]',
        'opencv-python',
        'sqlalchemy',
        'matplotlib',
        'pyvips',
        # dask packages
        'dask[dataframe]>=1.1.0',
        'distributed>=1.21.6',
        # large image sources
        'large-image[sources]',
        'girder-slicer-cli-web',
        # cli
        'ctk-cli',
    ],
    license='Apache Software License 2.0',
    keywords='histomicstk',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
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
