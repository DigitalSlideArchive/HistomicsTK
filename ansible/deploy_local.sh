#!/bin/bash

if [ $(lsb_release -cs) != "trusty" ]; then
  echo "This script will only work on Ubuntu 14.04"
  return 1
fi

unset PYTHONPATH
export GIRDER_EXEC_USER=`id -u -n`
ansible-playbook -i inventory/local deploy_local.yml
