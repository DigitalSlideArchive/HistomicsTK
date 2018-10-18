#!/bin/bash

if [[ -z "$HISTOMICS_SOURCE_FOLDER" ]]
then
  echo "Set environment variable HISTOMICS_SOURCE_FOLDER to HistomicsTK source directory"
  exit 1
fi

if [[ -z "$HISTOMICS_TESTDATA_FOLDER" ]]
then
  echo "Set environment variable HISTOMICS_TESTDATA_FOLDER to HistomicsTK source directory"
  exit 1
fi

$HISTOMICS_SOURCE_FOLDER/ansible/deploy_docker.py --mount $HISTOMICS_SOURCE_FOLDER:/opt/histomicstk/HistomicsTK/ --mount $HISTOMICS_TESTDATA_FOLDER:/data/ $@
