#!/usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################################
#  Copyright Kitware Inc.
#
#  Licensed under the Apache License, Version 2.0 ( the "License" );
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
##############################################################################

import hashlib
import json
import os
import shutil
import six
import sys
import subprocess
import tempfile
import unittest

from tests import base


TEST_DATA_DIR = os.path.join(os.environ['GIRDER_TEST_DATA_PREFIX'], 'plugins/HistomicsTK')


def setUpModule():
    # We need to start Girder to test endpoint creation
    base.enabledPlugins.append('HistomicsTK')
    base.startServer()


def tearDownModule():
    base.stopServer()


class CliResultsTest(unittest.TestCase):
    def _runTest(self, cli_args=(), cli_kwargs={}, outputs={}, contains=[],
                 excludes=[]):
        """
        Test a cli by calling subprocess.  Ensure that output files match a
        sha256 and stdout contains certain phrases and excludes other phrases.
        A value in the cli_args or cli_kwargs of 'tmp_<value>' will create a
        temporary file and use that instead.  The same value in the output
        list can be used to test the existence and contents of that file at the
        end of the process.

        :param cli_args: a tuple or list of args to format and pass to the cli.
        :param cli_kwargs: keyword arguments to format and pass to the cli.
            Each entry is passed as '--<key>=<value>'.
        :param outputs: a dictionary of expected output files and their hashes.
            The keys are the paths and the hash is the sha256 lowercase
            hexdigest.
        :param contains: a list of phrases that must be present in the stdout
            output of the cli.
        :param excludes: a list of phrases that must be not present in the
            stdout output of the cli.
        :returns: stdout from the test.
        """
        chunkSize = 256 * 1024

        stdout = ''
        cwd = os.environ.get('CLI_CWD')
        cmd = ['python', os.environ['CLI_LIST_ENTRYPOINT']]
        tmppath = tempfile.mkdtemp()
        try:
            cmd += [arg if not arg.startswith('tmp_') else
                    os.path.join(tmppath, arg) for arg in cli_args]
            cmd += ['--%s=%s' % (
                k, v if not v.startswith('tmp_') else os.path.join(tmppath, v))
                for k, v in six.iteritems(cli_kwargs)]
            process = subprocess.Popen(
                cmd, shell=False, stdout=subprocess.PIPE, cwd=cwd)
            stdout, stderr = process.communicate()
            self.assertEqual(process.returncode, 0)
            for outpath, outhash in six.iteritems(outputs):
                if outpath.startswith('tmp_'):
                    outpath = os.path.join(tmppath, outpath)
                self.assertTrue(os.path.isfile(outpath))
                sha = hashlib.sha256()
                with open(outpath, 'rb') as fptr:
                    while True:
                        data = fptr.read(chunkSize)
                        if not data:
                            break
                        sha.update(data)
                try:
                    self.assertEqual(outhash, sha.hexdigest())
                except Exception:
                    sys.stderr.write('File hash mismatch: %s\n' % outpath)
                    raise
            for entry in contains:
                self.assertIn(entry, stdout)
            for entry in excludes:
                self.assertNotIn(entry, stdout)
        except Exception:
            sys.stderr.write('CMD (cwd %s):\n%r\n' % (cwd, cmd))
            sys.stderr.write('STDOUT:\n%s\n' % stdout.rstrip())
            raise
        finally:
            shutil.rmtree(tmppath)
        return stdout

    def testListCLI(self):
        """
        Test that we can list clis, get help for each, and get the xml spec for
        each.
        """
        from girder.plugins.slicer_cli_web import rest_slicer_cli
        from girder.api.rest import Resource

        restResource = Resource()
        cli_args = ('--list_cli', )
        cli_list = self._runTest(cli_args, contains=['"NucleiDetection"'])
        cli_list = json.loads(cli_list)
        self.assertIn('NucleiDetection', cli_list)
        for cli in cli_list:
            cli_args = (cli, '--help')
            # Each cli's help must mention usage, its own name, and that you
            # can output xml
            self._runTest(cli_args, contains=['usage:', cli, '--xml'])
            cli_args = (cli, '--xml')
            # The xml output needs to have a tile and an executable tag
            xml = self._runTest(cli_args, contains=[
                '<executable>', '<title>', '</executable>', '</title>'])
            if '<' in xml:
                xml = xml[xml.index('<'):]
            try:
                rest_slicer_cli.genHandlerToRunDockerCLI(
                    'dockerimage', os.environ.get('CLI_CWD'), xml, restResource)
            except Exception:
                sys.stderr.write('Failed in generating endpoints for %s' % cli)
                raise

    def testNucleiDetectionDefaults(self):
        cli_args = (
            'NucleiDetection',
            os.path.join(TEST_DATA_DIR, 'Easy1.png'),
            'tmp_1.anot',
        )
        cli_kwargs = {
            'analysis_roi': '-1.0, -1.0, -1.0, -1.0',
        }
        self._runTest(cli_args, cli_kwargs, outputs={
            'tmp_1.anot': '02b240586412c87ad5cbf349b7c22f80f1df31eef54ed8ee4ad1fd3624a89fa2'
        })
