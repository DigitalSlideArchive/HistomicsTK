#!/bin/bash

if [ $(lsb_release -cs) != "xenial" ]; then
  echo "This script will only work on Ubuntu 16.04"
  return 1
fi

unset PYTHONPATH
export GIRDER_EXEC_USER=`id -u -n`
ansible-galaxy install -r requirements.yml -p roles/
ansible-playbook -i inventory/local deploy_local.yml
