#! /usr/bin/env python

import setuptools
from setuptools.command.test import test as TestCommand
import json
import sys


class Tox(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import tox
        errcode = tox.cmdline(self.test_args)
        sys.exit(errcode)

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
                 packages=['histomicstk'],
                 tests_require=['tox'],
                 cmdclass={'test': Tox})
