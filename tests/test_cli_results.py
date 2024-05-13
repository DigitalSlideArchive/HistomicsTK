import hashlib
import importlib
import json
import os
import tempfile
from pathlib import Path

import PIL.Image
import pytest

from .datastore import datastore


def _runCLITest(cli, args):
    from histomicstk.cli.utils import CLIArgumentParser
    module = importlib.import_module(f'histomicstk.cli.{cli}.{cli}')

    parentdir = Path(module.__file__).parent
    xmlfile = parentdir / f'{cli}.xml'
    # In our tox environment, the xml files aren't always copied
    while not xmlfile.exists():
        if parentdir.parent == parentdir:
            break
        parentdir = parentdir.parent
        xmlfile = parentdir / 'histomicstk' / 'cli' / cli / f'{cli}.xml'
    module.main(CLIArgumentParser(xmlfile).parse_args(args))


class TestNucleiDetection:

    def _runTest(self, args):
        with tempfile.TemporaryDirectory() as tmpdirname:
            outpath = os.path.join(tmpdirname, 'result.json')
            args = args + [outpath, '--scheduler=multithreading']
            _runCLITest('NucleiDetection', args)
            return json.load(open(outpath))

    @pytest.mark.parametrize(('filename', 'params'), [
        ('tcgaextract_rgb.tiff', []),
        ('tcgaextract_rgbmag.tiff', []),
        ('tcgaextract_ihergb_labeled.tiff', []),
        ('tcgaextract_ihergb_labeled.tiff', ['--frame', '1', '--invert_image', 'No']),
        ('tcgaextract_ihergb_labeled.tiff', ['--style', json.dumps({
            'bands': [
                {'palette': '#FF0000', 'framedelta': 3},
                {'palette': '#00FF00', 'framedelta': 4},
                {'palette': '#0000FF', 'framedelta': 5},
            ],
        })]),
        ('tcgaextract_ihergb_labeledmag.tiff', []),
        ('tcgaextract_ihergb_labeledmag.tiff', ['--frame', '1', '--invert_image', 'No']),
        ('tcgaextract_ihergb_labeledmag.tiff', ['--style', json.dumps({
            'bands': [
                {'palette': '#FF0000', 'framedelta': 3},
                {'palette': '#00FF00', 'framedelta': 4},
                {'palette': '#0000FF', 'framedelta': 5},
            ],
        })]),
    ])
    @pytest.mark.parametrize('roi', [[], ['--analysis_roi=1,1,3998,2998']])
    def test_detection(self, filename, params, roi):
        src = datastore.fetch(filename)
        annot = self._runTest([src] + params + roi)
        assert 2500 < len(annot['elements']) < 3500


class TestComputeNucleiFeatures:

    def _runTest(self, args):
        import pandas

        with tempfile.TemporaryDirectory() as tmpdirname:
            outpath1 = os.path.join(tmpdirname, 'result.csv')
            outpath2 = os.path.join(tmpdirname, 'result.json')
            args = args + [outpath1, outpath2, '--scheduler=multithreading']
            _runCLITest('ComputeNucleiFeatures', args)
            return pandas.read_csv(outpath1), json.load(open(outpath2))

    @pytest.mark.parametrize(('filename', 'params'), [
        ('tcgaextract_rgb.tiff', []),
    ])
    @pytest.mark.parametrize('roi', [[], ['--analysis_roi=1,1,3998,2998']])
    def test_detection(self, filename, params, roi):
        src = datastore.fetch(filename)
        feat, annot = self._runTest([src] + params + roi)
        assert 2500 < len(annot['elements']) < 3500
        assert len(feat) == len(annot['elements'])


class TestColorDeconvolution:

    def _runTest(self, args):
        with tempfile.TemporaryDirectory() as tmpdirname:
            outpath1 = os.path.join(tmpdirname, 'result1.png')
            outpath2 = os.path.join(tmpdirname, 'result2.png')
            outpath3 = os.path.join(tmpdirname, 'result3.png')
            args = args + [outpath1, outpath2, outpath3]
            _runCLITest('ColorDeconvolution', args)
            return PIL.Image.open(outpath1), PIL.Image.open(outpath2), PIL.Image.open(outpath3)

    def test_defaults(self):
        img1, img2, img3 = self._runTest([datastore.fetch('Easy1.png')])
        assert hashlib.sha256(img1.convert('L').tobytes()).hexdigest() in {
            'd46d8554ba626b8800cb7457d6a8d6c4e6915c75092fd783e0bfa2d4e85c06ec',
            '0ba7081bd71ca517e2beb96904b333485a93dc3eeab21590cca2bb4ab19b298e',
        }
        assert hashlib.sha256(img2.convert('L').tobytes()).hexdigest() in {
            'c48c3d21b674981ad38b902061b779cda5c605fda5e20a81894051e2a032e736',
            'adf0cf7246f395182e326b9798b8134e2ccfd021e607026cc97f2251be9c0f6d',
        }
        assert hashlib.sha256(img3.convert('L').tobytes()).hexdigest() in {
            '479066d2016122b2b60575f4e9abe201b4f79d8a894ce76191419e21c7539186',
            'd06317130f4a4aabd155c97f32cec4708ec3eec24d09d16daa2181f18b2051bb',
        }
