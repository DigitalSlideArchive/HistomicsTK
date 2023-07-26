#!/usr/bin/env python

import os
import subprocess

script = """
python --version && \\
pip install --upgrade pip && \\
pip install 'large_image[tiff,openslide,pil]' \\
  -f https://girder.github.io/large_image_wheels && \\
pip install histomicstk -f /wheels && \\
echo 'Test basic import of histomicstk' && \\
python -c 'import histomicstk' && \\
curl https://data.kitware.com/api/v1/file/5899dd6d8d777f07219fcb23/download -LJ -o tcga.svs && \\
export CLIPATH=`python -c 'import histomicstk,os;print(os.path.dirname( \\
  histomicstk.__file__))'`/cli && \\
python "$CLIPATH/NucleiDetection/NucleiDetection.py" tcga.svs sample.anot \\
  --analysis_roi="7000,7000,3000,3000" && \\
true"""

containers = [
    'python:3.7',
    'python:3.8',
    'python:3.9',
    'python:3.10',
    'python:3.11',
    'centos/python-38-centos7',
]

for container in containers:
    print('---- Testing in %s ----' % container)
    subprocess.check_call([
        'docker', 'run', '-v',
        '%s/wheels:/wheels' % os.path.dirname(os.path.realpath(__file__)),
        '--rm', container, 'sh', '-c', script])

# To test manually, run a container such as
#  docker run -v `pwd`/docs:/wheels --rm -it python:3.7 bash
# and then enter the script commands directly
