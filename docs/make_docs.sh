#!/usr/bin/env bash

set -e

cd "$(dirname $0)"
stat make_docs.sh

pushd ..
pip install . -r requirements-dev.txt --find-links https://girder.github.io/large_image_wheels
popd
# git clean -fxd .

make html

cp -r ../.circleci _build/html/.
