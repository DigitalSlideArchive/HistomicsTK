import json
import os
import tempfile
from pathlib import Path

import pytest

from .datastore import datastore


class TestNucleiDetection:

    def _runTest(self, args):
        from histomicstk.cli.NucleiDetection import NucleiDetection
        from histomicstk.cli.utils import CLIArgumentParser

        parentdir = Path(NucleiDetection.__file__).parent
        xmlfile = parentdir / 'NucleiDetection.xml'
        # In our tox environment, the xml files aren't always copied
        while not xmlfile.exists():
            if parentdir.parent == parentdir:
                break
            parentdir = parentdir.parent
            xmlfile = parentdir / 'histomicstk/cli/NucleiDetection/NucleiDetection.xml'
        with tempfile.TemporaryDirectory() as tmpdirname:
            outpath = os.path.join(tmpdirname, 'result.json')
            NucleiDetection.main(CLIArgumentParser(xmlfile).parse_args(
                args + [outpath, '--scheduler=multithreading']))
            return json.load(open(outpath))

    @pytest.mark.parametrize('filename,params', [
        ('tcgaextract_rgb.tiff', []),
        ('tcgaextract_rgbmag.tiff', []),
        ('tcgaextract_ihergb_labeled.tiff', []),
        ('tcgaextract_ihergb_labeled.tiff', ['--frame', '1', '--invert_image', 'No']),
        ('tcgaextract_ihergb_labeled.tiff', ['--style', json.dumps({
            'bands': [
                {'palette': '#FF0000', 'framedelta': 3},
                {'palette': '#00FF00', 'framedelta': 4},
                {'palette': '#0000FF', 'framedelta': 5}
            ]
        })]),
        ('tcgaextract_ihergb_labeledmag.tiff', []),
        ('tcgaextract_ihergb_labeledmag.tiff', ['--frame', '1', '--invert_image', 'No']),
        ('tcgaextract_ihergb_labeledmag.tiff', ['--style', json.dumps({
            'bands': [
                {'palette': '#FF0000', 'framedelta': 3},
                {'palette': '#00FF00', 'framedelta': 4},
                {'palette': '#0000FF', 'framedelta': 5}
            ]
        })]),
    ])
    @pytest.mark.parametrize('roi', [[], ['--analysis_roi=1,1,3998,2998']])
    def test_detection(self, filename, params, roi):
        src = datastore.fetch(filename)
        annot = self._runTest([src] + params + roi)
        assert 2000 < len(annot['elements']) < 3000
