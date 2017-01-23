#!/usr/bin/env python

import os

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
