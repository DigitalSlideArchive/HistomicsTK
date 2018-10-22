#!/bin/bash

# The following line does not work for symbolic link. Do not create a symbolic link
# of this file to run it.
HISTOMICS_SOURCE_FOLDER="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )/.."

if [[ -z "$HISTOMICS_TESTDATA_FOLDER" ]]
then
  echo "Set environment variable HISTOMICS_TESTDATA_FOLDER to HistomicsTK source directory"
  exit 1
fi

$HISTOMICS_SOURCE_FOLDER/ansible/deploy_docker.py --mount $HISTOMICS_SOURCE_FOLDER:/opt/histomicstk/HistomicsTK/ --mount $HISTOMICS_TESTDATA_FOLDER:/data/ $@
