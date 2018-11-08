#!/usr/bin/env python

import copy
import os
import sys


def adjust_ids(user_name):  # noqa
    """
    Make sure that the specified user matches the host user, the user's group
    matches the host group, and the user is a member of the host docker group
    and the docker's socket group (which are probably the same).

    :param user_name: the name of the user to check and modify.
    """
    user_uid = os.popen('id -u %s' % user_name).read().strip()
    user_gid = os.popen('id -g %s' % user_name).read().strip()
    # Get the group of /var/run/docker.sock
    sock_docker_gid = None
    sockpath = '/var/run/docker.sock'
    if os.path.exists(sockpath):
        # make the gid a string for later use
        sock_docker_gid = str(os.stat(sockpath).st_gid)
        escapedSockPath = "'%s'" % (sockpath.replace("'", "'\\''"))
        os.system('sudo chmod 777 %s' % escapedSockPath)
    # Get environment UID and GIDs.  These are not required to be set.
    host_uid = os.environ.get('HOST_UID')
    host_gid = os.environ.get('HOST_GID')
    host_docker_gid = os.environ.get('HOST_DOCKER_GID')
    # If the user's uid or gid differ from the hosts, we modify the /etc/passwd
    # file.  This fails if there is another user with the same uid as the
    # host's uid.
    user_entry = None
    if (host_uid and user_uid != host_uid) or (host_gid and user_gid != host_gid):
        # Parse the /etc/passwd file
        userlist = [{
            'parts': line.split(':', 4),
            'user': line.split(':', 1)[0],
            'uid': line.split(':', 3)[2],
            'gid': line.split(':', 4)[3],
            'home': line.split(':', 6)[5],
        } for line in open('/etc/passwd', 'rb').readlines()]
        isTaken = False
        for entry in userlist:
            if entry['user'] == user_name and entry['uid'] == user_uid and entry['gid'] == user_gid:
                user_entry = copy.deepcopy(entry)
                user_entry['parts'][0] = user_entry['user'] = 'user_' + user_uid
                entry['parts'][2] = host_uid or user_uid
                entry['parts'][3] = host_gid or user_gid
            elif entry['uid'] == host_uid:
                isTaken = True
        if isTaken:
            print('Error: The host user id is in use and does not match the current user id')
        else:
            if user_entry:
                userlist.append(user_entry)
            # Write the /etc/passwd file
            open('/etc/passwd.tmp', 'wb').write(''.join([
                ':'.join(entry['parts']) for entry in userlist]))
            os.rename('/etc/passwd.tmp', '/etc/passwd')
            os.system('pwconv')
    # Parse the /etc/group file
    grouplist = [{
        'parts': line.split(':'),
        'group': line.split(':', 1)[0],
        'gid': line.split(':', 3)[2],
        'users': line.split(':', 4)[3].strip().split(',')
    } for line in open('/etc/group', 'rb').readlines()]
    docker_entry = None
    # Add the user to existing groups
    for entry in grouplist:
        if entry['group'] == 'sudo' and user_entry and user_entry['user'] not in entry['users']:
            entry['parts'][3] = user_entry['user'] + (
                ',' if entry['parts'][3].strip() else '') + entry['parts'][3]
        if entry['group'] == 'docker':
            docker_entry = entry
        if entry['gid'] == host_gid and user_name not in entry['users']:
            entry['parts'][3] = user_name + (
                ',' if entry['parts'][3].strip() else '') + entry['parts'][3]
            entry['users'].append(user_name)
    # Create new groups as needed
    for gid in (host_gid, host_docker_gid, sock_docker_gid):
        if gid and not any([entry['gid'] == gid for entry in grouplist]):
            group = copy.deepcopy(docker_entry)
            group['gid'] = gid
            group['group'] = 'group_' + gid
            group['parts'][0] = group['group']
            group['parts'][2] = group['gid']
            group['parts'][3] = user_name + '\n'
            grouplist.append(group)
    # Write the /etc/group file
    open('/etc/group.tmp', 'wb').write(''.join([
        ':'.join(entry['parts']) for entry in grouplist]))
    os.rename('/etc/group.tmp', '/etc/group')
    os.system('grpconv')
    if host_uid != user_uid or host_gid != user_gid:
        os.system('rm -r /home/%s/.ansible' % user_name)
    if host_uid != user_uid:
        os.system('find / -xdev -uid %s -gid %s -exec chown -h %s:%s {} \+' % (
            user_uid, user_gid, host_uid or user_uid, host_gid or user_gid))
        os.system('find / -xdev -uid %s -exec chown -h %s {} \+' % (
            user_uid, host_uid or user_uid))
    if host_gid != user_gid:
        os.system('find / -xdev -gid %s -exec chgrp -h %s {} \+' % (
            user_gid, host_gid or user_gid))


def set_hosts():
    """
    Modify the /etc/hosts file to add a dockerhost entry and, if either the
    HOST_MONGO and HOST_RMQ environment variables are equal to 'true', also add
    the appropriate mongodb and rmq entries.
    """
    hostip = os.popen('netstat -nr').read().split('\n0.0.0.0')[1].strip().split()[0]

    hosts = [line.strip() for line in open('/etc/hosts').readlines()]

    changed = False
    if os.environ.get('HOST_MONGO') == 'true':
        if 'mongodb' not in [line.split()[-1] for line in hosts]:
            hosts.append('%s mongodb' % hostip)
            changed = True
    if os.environ.get('HOST_RMQ'):
        rmqhost = os.environ.get('HOST_RMQ')
        if rmqhost == 'true':
            rmqhost = hostip
        else:
            import socket
            rmqhost = socket.gethostbyname(rmqhost)
        if 'rmq' not in [line.split()[-1] for line in hosts]:
            hosts.append('%s rmq' % rmqhost)
            changed = True
    hosts.append('%s dockerhost' % hostip)
    changed = True
    if changed:
        open('/etc/hosts', 'wb').write('\n'.join(hosts) + '\n')


def set_tmp_root():
    """
    Add a specific directory to girder worker's tmp_root setting, and make sure
    that directory exists and has global rwx permissions.
    """
    tmpRoot = os.environ.get('GIRDER_WORKER_TMP_ROOT', '/tmp/girder_worker')
    escapedTmpRoot = "'%s'" % (tmpRoot.replace("'", "'\\''"))
    os.system('girder-worker-config set girder_worker tmp_root %s' % escapedTmpRoot)
    os.system('sudo mkdir -p %s' % escapedTmpRoot)
    os.system('sudo chmod a+rwx %s' % escapedTmpRoot)


if __name__ == '__main__':
    user_name = None
    tmp_root = False
    help = False
    for arg in sys.argv[1:]:
        if arg == 'tmp_root':
            tmp_root = True
        elif user_name is None:
            user_name = arg
        else:
            help = True
    if help:
        print """Adjust hosts, directories, and ids.

Syntax: set_environment.py [(user name)] [tmp_root]

  The /etc/hosts file is modified to add dockerhost and optionally mongodb and
rmq if the environment variables HOST_MONGO and/or HOST_RMQ are equal to
'true'.
  If a user name is specified, that user's UID and GID are set to match the
values in environment variables HOST_UID and HOST_GID.  The user is added to
the HOST_DOCKER_GID group and to whatever group owns /var/run/docker.sock.
  If tmp_root is specified, either GIRDER_WORKER_TMP_ROOT or /tmp/girder_worker
is set as girder_worker's tmp_root value, and the directory is created and set
to global access.

This must be run as a superuser.
"""
        sys.exit(0)
    set_hosts()
    if tmp_root:
        set_tmp_root()
    if user_name:
        adjust_ids(user_name)
