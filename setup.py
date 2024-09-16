import importlib
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


with open('README.rst') as readme_file:
    readme = readme_file.read()


setup(
    name='histomicstk',
    use_scm_version={'local_scheme': 'no-local-version',
                     'fallback_version': '0.0.0'},
    description='A Python toolkit for Histopathology Image Analysis',
    long_description=readme,
    long_description_content_type='text/x-rst',
    author='Kitware, Inc.',
    author_email='developers@digitalslidearchive.net',
    url='https://github.com/DigitalSlideArchive/HistomicsTK',
    packages=find_packages(exclude=['tests', '*_test*']),
    package_dir={
        'histomicstk': 'histomicstk',
    },
    include_package_data=True,
    install_requires=[
        'girder-client',
        # scientific packages
        'nimfa',
        'numpy',
        'scipy',
        'Pillow',
        'pandas',
        'scikit-image',
        'scikit-learn',
        'imageio',
        'shapely',
        'sqlalchemy',
        'matplotlib',
        'pyvips',
        # dask packages
        'dask[dataframe]',
        'distributed',
        # large image; for non-linux systems only install the PIL tile source
        # by default.
        'large-image[sources];sys.platform=="linux" or sys.platform=="linux2"',
        'large-image[common];sys.platform!="linux" and sys.platform!="linux2"',
        'large-image-converter;sys.platform=="linux" or sys.platform=="linux2"',
        'girder-slicer-cli-web',
        # cli
        'ctk-cli',
    ] + (
        # Only require opencv if it isn't installed.  This can allow alternate
        # forms to be in the environment (such as a headed form) without
        # causing conflicts
        [
            'opencv-python-headless',
        ] if not importlib.util.find_spec('cv2') else []
    ),
    license='Apache Software License 2.0',
    keywords='histomicstk',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    zip_safe=False,
    entry_points={
        'console_scripts': ['histomicstk = histomicstk.cli.__main__:main'],
    },
    python_requires='>=3.8',
)
