#!/bin/bash

unset PYTHONPATH
export GIRDER_EXEC_USER=`id -u -n`
ansible-playbook -i inventory/local deploy_local.yml
