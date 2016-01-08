#! /usr/bin/env python

import setuptools
import json

with open('README.rst') as f:
    readme = f.read()

with open('plugin.json') as f:
    pkginfo = json.load(f)

with open('LICENSE') as f:
    license = f.read()

setuptools.setup(name='histomicstk',
                 version=pkginfo['version'],
                 description=pkginfo['description'],
                 long_description=readme,
                 license=license,
                 url='https://github.com/DigitalSlideArchive/image_analysis',
                 packages=['histomicstk'])
