#!/usr/bin/env python

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
import io
import json
import os
import PIL.Image
import runpy
import shutil
import sys
import tempfile

from .datastore import datastore

base = None


def setUpModule():
    # We need to start Girder to test endpoint creation
    base.enabledPlugins.append('HistomicsTK')
    base.startServer()


def tearDownModule():
    base.stopServer()


class TestCliResults:
    def _runTest(self, cli_args=(), cli_kwargs=None, outputs=None,  # noqa
                 contains=None, excludes=None):
        """
        Test a cli by calling runpy.  Ensure that output files match a
        sha256 and stdout contains certain phrases and excludes other
        phrases.  A value in the cli_args or cli_kwargs of
        'tmp_<value>' will create a temporary file and use that
        instead.  The same value in the output list can be used to
        test the existence and contents of that file at the end of the
        process.  Note that stdout capturing has limitations -- only
        output written respecting the current value of sys.stdout is
        captured.

        :param cli_args: a tuple or list of args to format and pass to the cli.
        :param cli_kwargs: keyword arguments to format and pass to the cli.
            Each entry is passed as '--<key>=<value>'.
        :param outputs: a dictionary of expected output files and checks to
            perform on them.  The keys are the file paths.  The values are
            dictionaries that may contain:
                'hash': a sha256 lowercase hexdigest that must match the entire
                    file.
                'contains': a list of phrases that must be present in the first
                    256 kb of the file.
        :param contains: a list of phrases that must be present in the stdout
            output of the cli.
        :param excludes: a list of phrases that must be not present in the
            stdout output of the cli.
        :returns: stdout from the test.

        """
        cli_kwargs = {} if cli_kwargs is None else cli_kwargs
        outputs = {} if outputs is None else outputs
        contains = [] if contains is None else contains
        excludes = [] if excludes is None else excludes

        chunkSize = 256 * 1024

        cwd = os.environ.get('CLI_CWD')
        tmppath = tempfile.mkdtemp()
        try:
            cmd = [arg if not arg.startswith('tmp_') else
                   os.path.join(tmppath, arg) for arg in cli_args]
            cmd += ['--%s=%s' % (
                k, v if not v.startswith('tmp_') else os.path.join(tmppath, v))
                for k, v in cli_kwargs.items()]
            stdout = ''
            try:
                old_sys_argv = sys.argv[:]
                old_stdout, old_stderr = sys.stdout, sys.stderr
                old_cwd = os.getcwd()
                sys.argv[:] = cmd
                sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
                os.chdir(cwd)
                runpy.run_path(
                    # If passed a Python file, run it directly
                    cli_args[0] if cli_args[0].endswith('.py') else
                    os.path.join(cwd, cli_args[0], cli_args[0] + '.py'),
                    run_name='__main__',
                )
            except SystemExit as e:
                assert e.code in {0, None}
            finally:
                stdout = sys.stdout.getvalue()
                sys.argv[:] = old_sys_argv
                sys.stdout, sys.stderr = old_stdout, old_stderr
                os.chdir(old_cwd)
            for entry in contains:
                assert entry in stdout
            for entry in excludes:
                assert entry not in stdout
            for outpath, options in outputs.items():
                if outpath.startswith('tmp_'):
                    outpath = os.path.join(tmppath, outpath)
                assert os.path.isfile(outpath)
                if 'hash' in options:
                    sha = hashlib.sha256()
                    # For png test files, take a hash of the decoded image
                    # rather than the file.
                    if outpath.endswith('.png'):
                        sha.update(PIL.Image.open(outpath).convert('RGBA').tobytes())  # noqa
                    else:
                        with open(outpath, 'rb') as fptr:
                            while True:
                                data = fptr.read(chunkSize)
                                if not data:
                                    break
                                sha.update(data)
                    hashval = sha.hexdigest()
                    try:
                        assert options['hash'] == hashval
                    except Exception:
                        sys.stderr.write('File hash mismatch: %s\n' % outpath)
                        raise
                if 'contains' in options:
                    data = open(outpath).read(chunkSize)
                    for entry in options['contains']:
                        assert entry in data
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
        cli_args = (os.environ['CLI_LIST_ENTRYPOINT'], '--list_cli',)
        cli_list = self._runTest(cli_args, contains=['"NucleiDetection"'])
        cli_list = json.loads(cli_list)
        assert 'NucleiDetection' in cli_list
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
            os.path.join(TEST_DATA_DIR, 'TCGA-02-0010-01Z-00-DX4.07de2e55-a8fe-40ee-9e98-bcb78050b9f7-crop.tif'),   # noqa
            'tmp_1.anot',
        )

        # test without ROI
        cli_kwargs = {
            'analysis_roi': '-1.0, -1.0, -1.0, -1.0',
        }
        self._runTest(cli_args, cli_kwargs, outputs={
            'tmp_1.anot': {
                'contains': ['elements'],
                # 'hash': '02b240586412c87ad5cbf349b7c22f80f1df31eef54ed8ee4ad1fd3624a89fa2',  # noqa
            },
        })

        # test with ROI
        cli_kwargs = {
            'analysis_roi': '0.0, 200.0, 512.0, 512.0'
        }
        self._runTest(cli_args, cli_kwargs, outputs={
            'tmp_1.anot': {
                'contains': ['elements'],
                # 'hash': '02b240586412c87ad5cbf349b7c22f80f1df31eef54ed8ee4ad1fd3624a89fa2',  # noqa
            },
        })

    def testColorDeconvolutionDefaults(self):
        self._runTest(
            cli_args=[
                'ColorDeconvolution',
                datastore.fetch('Easy1.png'),
            ] + [f'tmp_out_{i}.png' for i in (1, 2, 3)],
            outputs={
                'tmp_out_1.png': dict(
                    hash='b91d961b7eba8c02a1c067da91fced315fa3922db73f489202a296f4c2304b94',  # noqa
                ),
            },
        )
