#!/usr/bin/env bash

set -e

cd "$(dirname $0)"

# Remove old docs if they exist
rm -fr _build

# Make the docs
make html

# Tell circleci not to run CI on the docs output
cp -r ../.circleci _build/html/.
