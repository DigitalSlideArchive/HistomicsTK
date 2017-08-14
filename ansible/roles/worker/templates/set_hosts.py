#!/usr/bin/env python

import os
import sys

hostip = os.popen('netstat -nr').read().split('\n0.0.0.0')[1].strip().split()[0]

hosts = [line.strip() for line in open('/etc/hosts').readlines()]

changed = False
if os.environ.get('HOST_MONGO') == 'true':
    if 'mongodb' not in [line.split()[-1] for line in hosts]:
        hosts.append('%s mongodb' % hostip)
        changed = True
if os.environ.get('HOST_RMQ') == 'true':
    if 'rmq' not in [line.split()[-1] for line in hosts]:
        hosts.append('%s rmq' % hostip)
        changed = True
hosts.append('%s dockerhost' % hostip)
changed = True
if changed:
    open('/etc/hosts', 'wb').write('\n'.join(hosts) + '\n')


tmpRoot = os.environ.get('GIRDER_WORKER_TMP_ROOT', '/tmp/girder_worker')
escapedTmpRoot = "'%s'" % (tmpRoot.replace("'", "'\\''"))
os.system('girder-worker-config set girder_worker tmp_root %s' % escapedTmpRoot)
os.system('sudo mkdir -p %s' % escapedTmpRoot)
os.system('sudo chmod a+rwx %s' % escapedTmpRoot)


# Make sure we are a member of the group that docker's socket file belongs to

sockpath = '/var/run/docker.sock'
if os.path.exists(sockpath):
    docker_gid = os.stat(sockpath).st_gid
    if not os.popen('getent group %s' % docker_gid).read():
        # make sure there is no group called dockerhost
        os.system('sudo delgroup dockerhost')
        # create a group called dockerhost with the group id from the host
        os.system('sudo addgroup --gid=%s dockerhost' % docker_gid)
        # add the current user to it
        os.system('sudo usermod -aG %s `id -u -n`' % docker_gid)
        # add the worker user to it
        os.system('sudo usermod -aG %s {{ worker_exec_user }}' % docker_gid)
        # Force a restart so the group is available
        sys.exit(1)
