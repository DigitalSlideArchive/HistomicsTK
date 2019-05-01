#!/usr/bin/env bash

set -e

docker build --force-rm --tag dsarchive/histomicstk_wheels -f Dockerfile-wheels .
mkdir -p wheels
docker run -v `pwd`/wheels:/opt/mount --rm --entrypoint bash dsarchive/histomicstk_wheels -c 'cp /io/wheelhouse/histomicstk*many* /opt/mount/. && chown '`id -u`':'`id -g`' /opt/mount/*.whl'
# python make_index.py
ls -l wheels

