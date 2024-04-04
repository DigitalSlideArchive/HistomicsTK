#!/usr/bin/env bash

set -e

docker build --force-rm --tag dsarchive/histomicstk_wheels --build-arg "CIRCLE_BRANCH=$CIRCLE_BRANCH" -f Dockerfile-wheels .
# mkdir -p wheels
# docker run -v `pwd`/wheels:/opt/mount --rm --entrypoint bash dsarchive/histomicstk_wheels -c 'cp /io/wheelhouse/histomicstk*many* /opt/mount/. && chown '`id -u`':'`id -g`' /opt/mount/*.whl'
# ls -l wheels
docker create --name wheels dsarchive/histomicstk_wheels
docker cp wheels:/io/wheels .
docker rm wheels

python -m pip install scikit-build
python setup.py sdist
