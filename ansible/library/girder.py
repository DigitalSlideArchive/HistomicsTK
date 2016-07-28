#!/usr/bin/python
# -*- coding: utf-8 -*-

###############################################################################
#  Copyright Kitware Inc.
#
#  Licensed under the Apache License, Version 2.0 ( the "License" );
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

import json
import os
from inspect import getmembers, ismethod, getargspec

# Ansible's module magic requires this to be
# 'from ansible.module_utils.basic import *' otherwise it will error out. See:
# https://github.com/ansible/ansible/blob/v1.9.4-1/lib/ansible/module_common.py#L41-L59
# For more information on this magic. For now we noqa to prevent flake8 errors
from ansible.module_utils.basic import *  # noqa

try:
    from girder_client import GirderClient, AuthenticationError, HttpError
    HAS_GIRDER_CLIENT = True
except ImportError:
    HAS_GIRDER_CLIENT = False


__version__ = "0.2.3"

DOCUMENTATION = '''
---
module: girder
author: "Chris Kotfila (chris.kotfila@kitware.com)
version_added: "0.1"
short_description: A module that wraps girder_client
requirements: [ girder_client==1.1.0 ]
description:
   - Manage a girder instance using the RESTful API
options:
    host:
        required: false
        default: 'localhost'
        description:
            - domain or IP of the host running girder
    port:
        required: false
        default: '80' for http, '443' for https
        description:
            - port the girder instance is running on

    apiRoot:
        required: false
        default: '/api/v1'
        description:
            - path on server corresponding to the root of Girder REST API

    apiUrl:
        required: false
        default: None
        description:
            - full URL base of the girder instance API
    apiKey:
        required: false
        default: None
        description:
            - pass in an apiKey instead of username/password


    scheme:
        required: false
        default: 'http'
        description:
            - A string containing the scheme for the Girder host

    dryrun:
        required: false
        default: None (passed through)
        description:
            - See GirderClient.__init__()

    blacklist:
        required: false
        default: None (passed through)
        description:
            - See GirderClient.__init__()

    username:
        required: true
        description:
            - Valid username for the system
            - Required with password
            - must be specified if 'token' is not specified
                - (See note on 'user')

    password:
        required: true
        description:
            - Valid password for the system
            - Required with username
            - must be specified if 'token' is not specified
                - (See note on 'user')
    token:
        required: true
        description:
            - A girder client token
            - Can be retrieved by accessing the accessing the 'token' attribute
              from a successfully authenticated call to girder in a previous
              task.
            - Required if 'username' and 'password' are not specified
                - (See note on 'user')
    state:
        required: false
        default: "present"
        choices: ["present", "absent"]
        description:
            - Used to indicate the presence or absence of a resource
              - e.g.,  user, plugin, assetstore

    user:
        required: false
        description:
            - If using the 'user' task, you are NOT REQUIRED to pass in a
              'username' & 'password',  or a 'token' attributes. This is because
              the first user created on an fresh install of girder is
              automatically made an administrative user. Once you are certain
              you have an admin user you should use those credentials in all
              subsequent tasks that use the 'user' task.

            - Takes a mapping of key value pairs
              options:
                  login:
                      required: true
                      description:
                          - The login name of the user
                  password:
                      required: true
                      description:
                          - The password of the user

                  firstName:
                      required: false
                      default: pass through to girder client
                      description:
                          - The first name of the user

                  lastName:
                      required: false
                      default: pass through to girder client
                      description:
                          - The last name of the user
                  email:
                      required: false
                      default: pass through to girder client
                      description:
                          - The email of the user
                  admin:
                      required: false
                      default: false
                      description:
                          - If true,  make the user an administrator.


    plugin:
        required: false
        description:
            - Specify what plugins should be activated (state: present)
              or deactivated (state: absent).
            - Takes a list of plugin names,  incorrect names are silently
              ignored

    assetstore:
        required: false
        description:
            - Specifies an assetstore
            - Takes many options depending on 'type'
              options:
                  name:
                      required: true
                      description:
                          - Name of the assetstore
                  type:
                      required: true
                      choices: ['filesystem', 'gridfs', 's3', 'hdfs']
                      description:
                          - Currently only 'filesystem' has been tested
                  readOnly:
                      required: false
                      default: false
                      description:
                          - Should the assetstore be read only?
                  current:
                      required: false
                      default: false
                      description:
                          - Should the assetstore be set as the current
                            assetstore?

              options (filesystem):
                  root:
                      required: true
                      description:
                          -  Filesystem path to the assetstore

              options (gridfs) (EXPERIMENTAL):
                   db:
                       required: true
                       description:
                           - database name
                   mongohost:
                       required: true
                       description:
                           - Mongo host URI

                   replicaset:
                       required: false
                       default: ''
                       description:
                           - Replica set name

              options (s3) (EXPERIMENTAL):
                   bucket:
                       required: true
                       description:
                           - The S3 bucket to store data in

                   prefix:
                       required: true
                       description:
                           - Optional path prefix within the bucket under which
                             files will be stored

                   accessKeyId:
                       required: true
                       description:
                           - the AWS access key ID to use for authentication

                   secret:
                       required: true
                       description:
                           - the AWS secret key to use for authentication

                   service:
                       required: false
                       default: s3.amazonaws.com
                       description:
                           - The S3 service host (for S3 type)
                           - This can be used to specify a protocol and port
                             -  use the form [http[s]://](host domain)[:(port)]
                           - Do not include the bucket name here

              options (hdfs) (EXPERIMENTAL):
                   host:
                       required: true
                       description:
                           - None
                   port:
                       required: true
                       description:
                           - None
                   path:
                       required: true
                       description:
                           - None
                   user:
                       required: true
                       description:
                           - None
                   webHdfsPort
                       required: true
                       description:
                           - None

    group:
        required: false
        description:
            - Create a group with pre-existing users
        options:
            name:
                required: true
                description:
                    - Name of the group

            description:
                required: false
                description:
                    - Description of the group
            users:
                required: false
                type: list
                description:
                    - List of dicts with users login and their level
                options:
                    login:
                        required: true
                        description:
                            - the login name
                    type:
                        required: true
                        choices: ["member", "moderator", "admin"]
                        description:
                            - Access level for that user in the group

    collection:
        required: false
        description:
            - Create a collection
        options:
            name:
                required: true
                description:
                    - Name of the collection

            description:
                required: false
                description:
                    - Description of the collection

            folders:
                required: false
                description:
                    - A list of folder options
                    - Specified by the 'folder' option to the girder module
                    - (see 'folder:')
            access:
                required: false
                description:
                    - Set the access for the collection/folder
                options:
                    users:
                        required: false
                        description:
                            - list of login/type arguments
                            - login is a user login
                            - type is one of 'admin', 'moderator', 'member'
                    groups:
                        required: false
                        description:
                            - list of name/type arguments
                            - name is a group name
                            - type is one of 'admin', 'moderator', 'member'

    folder:
        required: false
        description:
            - Create a folder
        options:
            name:
                required: true
                description:
                    - Name of the folder

            description:
                required: false
                description:
                    - Description of the folder
            parentType:
                required: true
                choices: ["user", "folder", "collection"]
                description:
                    - The type of the parent
            parentId:
                required: true
                description:
                    - The ID of the parent collection
            folders:
                required: false
                description:
                    - A list of folder options
                    - Specified by the 'folder' option to the girder module
                    - (see 'folder:')
            access:
                required: false
                description:
                    - Set the access for the collection/folder
                options:
                    users:
                        required: false
                        description:
                            - list of login/type arguments
                            - login is a user login
                            - type is one of 'admin', 'moderator', 'member'
                    groups:
                        required: false
                        description:
                            - list of name/type arguments
                            - name is a group name
                            - type is one of 'admin', 'moderator', 'member'

    item:
        required: false
        description:
            - Create a item
        options:
            name:
                required: true
                description:
                    - Name of the item

            description:
                required: false
                description:
                    - Description of the item
            folderId:
                required: true
                description:
                    - The ID of the parent collection

     files:
        required: false
        description:
            - Uploads a list of files to an item
        options:
            itemId:
                required: true
                description:
                    - the parent item for the file
            sources:
                required: true
                description:
                    - list of local file paths
                    - files will be uploaded to the item

'''

EXAMPLES = '''


#############
# Example using 'user'
###


# Ensure "admin" user exists
- name: Create 'admin' User
  girder:
    user:
      firstName: "Chris"
      lastName: "Kotfila"
      login: "admin"
      password: "letmein"
      email: "chris.kotfila@kitware.com"
      admin: yes
    state: present

# Ensure a 'foobar' user exists
- name: Create 'foobar' User
  girder:
    username: "admin"
    password: "letmein"
    user:
      firstName: "Foo"
      lastName: "Bar"
      login: "foobar"
      password: "foobarbaz"
      email: "foo.bar@kitware.com"
      admin: yes
    state: present

# Remove the 'foobar' user
- name: Remove 'foobar' User
  username: "admin"
  password: "letmein"
  girder:
    user:
      login: "foobar"
      password: "foobarbaz"
    state: absent

############
# Examples using Group
#

# Create an 'alice' user
- name: Create 'alice' User
  girder:
    port: 8080
    username: "admin"
    password: "letmein"
    user:
      firstName: "Alice"
      lastName: "Test"
      login: "alice"
      password: "letmein"
      email: "alice.test@kitware.com"
    state: present

# Create a 'bill' user
- name: Create 'bill' User
  girder:
    port: 8080
    username: "admin"
    password: "letmein"
    user:
      firstName: "Bill"
      lastName: "Test"
      login: "bill"
      password: "letmein"
      email: "bill.test@kitware.com"
    state: present

# Create a 'chris' user
- name: Create 'chris' User
  girder:
    port: 8080
    username: "admin"
    password: "letmein"
    user:
      firstName: "Chris"
      lastName: "Test"
      login: "chris"
      password: "letmein"
      email: "chris.test@kitware.com"
    state: present

- name: Create a test group with users
  girder:
    port: 8080
    username: "admin"
    password: "letmein"
    group:
      name: "Test Group"
      description: "Basic test group"
      users:
        - login: alice
          type: member
        - login: bill
          type: moderator
        - login: chris
          type: admin
    state: present

# Remove Bill from the group,
# Note that 'group' list is idempotent - it describes the desired state

- name: Remove bill from group
  girder:
    port: 8080
    username: "admin"
    password: "letmein"
    group:
      name: "Test Group"
      description: "Basic test group"
      users:
        - login: alice
          type: member
        - login: chris
          type: admin
    state: present

#############
# Example using 'plugins'
###

# To enable or disable all plugins you may pass the "*"
# argument.  This does not (yet) support arbitrary regexes
- name: Disable all plugins
  girder:
    username: "admin"
    password: "letmein"
    plugins: "*"
    state: absent

- name: Enable thumbnails plugin
  girder:
    username: "admin"
    password: "letmein"
    port: 8080
    plugins:
      - thumbnails
    state: present

# Note that 'thumbnails'  is still enabled from the previous task,
# the 'plugins' task ensures that plugins are enabled or disabled,
# it does NOT define the complete list of enabled or disabled plugins.
- name: Ensure jobs and gravatar plugins are enabled
  girder:
    username: "admin"
    password: "letmein"
    plugins:
      - jobs
      - gravatar
    state: present



############
# Filesystem Assetstore Tests
#

- name: Create filesystem assetstore
  girder:
    username: "admin"
     password: "letmein"
     assetstore:
       name: "Temp Filesystem Assetstore"
       type: "filesystem"
       root: "/data/"
       current: true
     state: present

- name: Delete filesystem assetstore
  girder:
    username: "admin"
    password: "letmein"
    assetstore:
      name: "Temp Filesystem Assetstore"
      type: "filesystem"
      root: "/tmp/"
    state: absent


############
# Examples using collections, folders, items and files
#

# Creates a test collection called "Test Collection"
- name: Create collection
  girder:
    port: 8080
    username: "admin"
    password: "letmein"
    collection:
      name: "Test Collection"
      description: "A test collection"
  register: test_collection

# Creates a folder called "test folder" under "Test Collection"
- name: Create folder
  girder:
    port: 8080
    username: "admin"
    password: "letmein"
    folder:
      parentType: "collection"
      parentId: "{{test_collection['gc_return']['_id'] }}"
      name: "test folder"
      description: "A test folder"
  register: test_folder

# Creates an item called "test item" under "test folder"
- name: Create an item
  girder:
    port: 8080
    username: "admin"
    password: "letmein"
    item:
      folderId: "{{test_folder['gc_return']['_id'] }}"
      name: "test item"
      description: "A test item"
  register: test_item

# Upload files on the localhost at /tmp/data/test1.txt and
# /tmp/data/test2.txt to the girder instance under the item
# "test item"
# Note:  the list is idempotent and will remove files that are
# not listed under the item. Files are checked for both name
# and size to determine if they should be updated.
- name: Upload files
  girder:
    port: 8080
    username: "admin"
    password: "letmein"
    files:
      itemId: "{{ test_item['gc_return']['_id'] }}"
      sources:
        - /tmp/data/test1.txt
        - /tmp/data/test2.txt
  register: retval


############
# Examples Using collection/folder hierarchy
#

- name: Create collection with a folder and a subfolder
  girder:
    port: 8080
    username: "admin"
    password: "letmein"
    collection:
      name: "Test Collection"
      description: "A test collection"
      folders:
        - name: "test folder"
          description: "A test folder"
          folders:
            - name: "test subfolder"
            - name: "test subfolder 2"
  register: test_collection



############
# Examples Setting access to files/folders
#


- name: Create collection with access
  girder:
    port: 8080
    username: "admin"
    password: "letmein"
    collection:
      name: "Test Collection"
      description: "A test collection"
      public: no
      access:
        users:
          - login: alice
            type: admin
          - login: chris
            type: member
  register: test_collection


- name: Add group to Test Collection
  girder:
    port: 8080
    username: "admin"
    password: "letmein"
    collection:
      name: "Test Collection"
      description: "A test collection"
      public: no
      access:
        users:
          - login: alice
            type: admin
          - login: bill
            type: moderator
          - login: chris
            type: member
        groups:
          - name: Test Group
            type: member
  register: test_collection

- name: Add Test Folder with access
  girder:
    port: 8080
    username: "admin"
    password: "letmein"
    folder:
      parentType: "collection"
      parentId: "{{test_collection['gc_return']['_id'] }}"
      name: "test folder"
      description: "A test folder"
      access:
        users:
          - login: bill
            type: admin
        groups:
          - name: Test Group
            type: member
  register: test_folder



############
# Examples using get
#


# Get my info
- name: Get users from http://localhost:80/api/v1/users
  girder:
    username: 'admin'
    password: 'letmein'
    get:
      path: "users"
    register: ret_val

# Prints debugging messages with the emails of the users
# From the last task by accessing 'gc_return' of the registered
# variable 'ret_val'
- name: print emails of users
  debug: msg="{{ item['email'] }}"
  with_items: "{{ ret_val['gc_return'] }}"


#############
# Advanced usage
#

# Supports get, post, put, delete methods,  but does
# not guarantee idempotence on these methods!

- name: Restart the server
  girder:
    username: "admin"
    password: "letmein"
    put:
      path: "system/restart"

# An example of posting an item to Girder
# Note that this is NOT idempotent. Running
# multiple times will create "An Item", "An Item (1)",
# "An Item (2)", etc..

- name: Get Me
  girder:
    username: "admin"
    password: "letmein"
    get:
      path: "user/me"
  register: ret

# Show use of 'token' for subsequent authentication
- name: Get my public folder
  girder:
    token: "{{ ret['token'] }}"
    get:
      path: "folder"
      parameters:
        parentType: "user"
        parentId: "{{ ret['gc_return']['_id'] }}"
        text: "Public"
  register: ret


- name: Post an item to my public folder
  girder:
    host: "data.kitware.com"
    scheme: 'https'
    token: "{{ ret['token'] }}"
    post:
      path: "item"
      parameters:
        folderId: "{{ ret['gc_return'][0]['_id'] }}"
        name: "An Item"


'''


def class_spec(cls, include=None):
    include = include if include is not None else []

    for fn, method in getmembers(cls, predicate=ismethod):
        if fn in include:
            spec = getargspec(method)
            # Note: must specify the kind of data we accept
            #       In all most all cases this will be a dict
            #       where variable names become keys used in yaml
            #       but if we have a vararg then we need to set
            #       this to a list.
            kind = 'dict' if spec.varargs is None else 'list'

            # spec.args[1:] so we don't include 'self'
            params = spec.args[1:]
            d = len(spec.defaults) if spec.defaults is not None else 0
            r = len(params) - d

            yield (fn, {"required": params[:r],
                        "optional": params[r:],
                        "type": kind})


class Resource(object):
    known_resources = ['collection', 'folder', 'item', 'group']

    def __init__(self, client, resource_type):
        self._resources = None
        self._resources_by_name = None
        self.client = client

        if resource_type in self.known_resources:
            self.resource_type = resource_type
        else:
            raise Exception("{} is an unknown resource!".format(resource_type))

    @property
    def resources(self):
        if self._resources is None:
            self._resources = {r['_id']: r for r
                               in self.client.get(self.resource_type)}
        return self._resources

    @property
    def resources_by_name(self):
        if self._resources_by_name is None:
            self._resources_by_name = {r['name']: r
                                       for r in self.resources.values()}

        return self._resources_by_name

    def __apply(self, _id, func, *args, **kwargs):
        if _id in self.resources.keys():
            ret = func("{}/{}".format(self.resource_type, _id),
                       *args, **kwargs)
            self.client.changed = True
        return ret

    def id_exists(self, _id):
        return _id in self.resources.keys()

    def name_exists(self, _name):
        return _name in self.resources_by_name.keys()

    def create(self, body, **kwargs):
        try:
            ret = self.client.post(self.resource_type, body, **kwargs)
            self.client.changed = True
        except HttpError as htErr:
            try:
                # If we can't create the item,  try and return
                # The item with the same name
                ret = self.resource_by_name[kwargs['name']]
            except KeyError:
                raise htErr
        return ret

    def read(self, _id):
        return self.resources[_id]

    def read_by_name(self, name):
        return self.resources_by_name[name]['_id']

    def update(self, _id, body, **kwargs):
        if _id in self.resources:
            current = self.resources[_id]
            # if body is a subset of current we don't actually need to update
            if set(body.items()) <= set(current.items()):
                return current
            else:
                return self.__apply(_id, self.client.put, body, **kwargs)
        else:
            raise Exception("{} does not exist!".format(_id))

    def update_by_name(self, name, body, **kwargs):
        return self.update(self.resources_by_name[name]['_id'],
                           body, **kwargs)

    def delete(self, _id):
        return self.__apply(_id, self.client.delete)

    def delete_by_name(self, name):
        try:
            return self.delete(self.resources_by_name[name]['_id'])
        except KeyError:
            return {}


class AccessMixin(object):

    def get_access(self, _id):
        return self.client.get("{}/{}/access"
                               .format(self.resource_type, _id))

    def put_access(self, _id, access, public=True):
        current_access = self.get_access(_id)

        if set([tuple(u.values()) for u in access['users']]) ^ \
           set([(u["id"], u['level']) for u in current_access['users']]):
            self.client.changed = True

        if set([tuple(g.values()) for g in access['groups']]) ^ \
           set([(u["id"], u['level']) for u in current_access['groups']]):
            self.client.changed = True

        return self.client.put("{}/{}/access"
                               .format(self.resource_type, _id),
                               dict(access=json.dumps(access),
                                    public="true" if public else "false"))


class CollectionResource(AccessMixin, Resource):
    def __init__(self, client):
        super(CollectionResource, self).__init__(client, "collection")


class GroupResource(Resource):
    def __init__(self, client):
        super(GroupResource, self).__init__(client, "group")


class FolderResource(AccessMixin, Resource):
    def __init__(self, client, parentType, parentId):
        super(FolderResource, self).__init__(client, "folder")
        self.parentType = parentType
        self.parentId = parentId

    @property
    def resources(self):
        if self._resources is None:
            self._resources = {r['_id']: r for r
                               in self.client.get(self.resource_type, {
                                   "parentType": self.parentType,
                                   "parentId": self.parentId
                               })}
            # parentType is stored as parrentCollection in database
            # We need parentType to be available so we can do set
            # comparison to check if we are updating parentType (e.g.
            # Moving a subfolder from a folder to a collection)
            for _id in self._resources.keys():
                self._resources[_id]['parentType'] = \
                    self._resources[_id]['parentCollection']
        return self._resources


class ItemResource(Resource):
    def __init__(self, client, folderId):
        super(ItemResource, self).__init__(client, "item")
        self.folderId = folderId

    @property
    def resources(self):
        if self._resources is None:
            self._resources = {r['_id']: r for r
                               in self.client.get(self.resource_type, {
                                   "folderId": self.folderId
                               })}
        return self._resources


class GirderClientModule(GirderClient):

    # Exclude these methods from both 'raw' mode
    _include_methods = ['get', 'put', 'post', 'delete', 'patch',
                        'plugins', 'user', 'assetstore',
                        'collection', 'folder', 'item', 'files',
                        'group']

    _debug = True

    def exit(self):
        if not self._debug:
            del self.message['debug']

        self.module.exit_json(changed=self.changed, **self.message)

    def fail(self, msg):
        self.module.fail_json(msg=msg)

    def __init__(self):
        self.changed = False
        self.message = {"msg": "Success!", "debug": {}}

        self.spec = dict(class_spec(self.__class__,
                                    GirderClientModule._include_methods))
        self.required_one_of = self.spec.keys()

        # Note: if additional types are added o girder this will
        # have to be updated!
        self.access_types = {"member": 0, "moderator": 1, "admin": 2}

    def __call__(self, module):
        self.module = module

        super(GirderClientModule, self).__init__(
            **{p: self.module.params[p] for p in
               ['host', 'port', 'apiRoot', 'apiUrl',
                'scheme', 'dryrun', 'blacklist']
               if module.params[p] is not None})
        # If a username and password are set
        if self.module.params['username'] is not None:
            try:
                self.authenticate(
                    username=self.module.params['username'],
                    password=self.module.params['password'])

            except AuthenticationError:
                self.fail("Could not Authenticate!")

        elif self.module.params['apiKey'] is not None:
            try:
                self.authenticate(
                    apiKey=self.module.params['apiKey'])

            except AuthenticationError:
                self.fail("Could not Authenticate!")

        # If a token is set
        elif self.module.params['token'] is not None:
            self.token = self.module.params['token']

        # Else error if we're not trying to create a user
        elif self.module.params['user'] is None:
            self.fail("Must pass in either username & password, "
                      "or a valid girder_client token")

        self.message['token'] = self.token

        for method in self.required_one_of:
            if self.module.params[method] is not None:
                self.__process(method)
                self.exit()

        self.fail("Could not find executable method!")

    def __process(self, method):
        # Parameters from the YAML file
        params = self.module.params[method]
        # Final list of arguments to the function
        args = []
        # Final list of keyword arguments to the function
        kwargs = {}

        if isinstance(params, dict):
            for arg_name in self.spec[method]['required']:
                if arg_name not in params.keys():
                    self.fail("%s is required for %s" % (arg_name, method))
                args.append(params[arg_name])

            for kwarg_name in self.spec[method]['optional']:
                if kwarg_name in params.keys():
                    kwargs[kwarg_name] = params[kwarg_name]

        elif isinstance(params, list):
            args = params
        else:
            args = [params]

        ret = getattr(self, method)(*args, **kwargs)

        self.message['debug']['method'] = method
        self.message['debug']['args'] = args
        self.message['debug']['kwargs'] = kwargs
        self.message['debug']['params'] = params

        self.message['gc_return'] = ret

    def files(self, itemId, sources=None):
        ret = {"added": [],
               "removed": []}

        files = self.get("item/{}/files".format(itemId))

        if self.module.params['state'] == 'present':

            file_dict = {f['name']: f for f in files}

            source_dict = {os.path.basename(s): {
                "path": s,
                "name": os.path.basename(s),
                "size": os.path.getsize(s)} for s in sources}

            source_names = set([(s['name'], s['size'])
                                for s in source_dict.values()])

            file_names = set([(f['name'], f['size'])
                              for f in file_dict.values()])

            for n, _ in (file_names - source_names):
                self.delete("file/{}".format(file_dict[n]['_id']))
                ret['removed'].append(file_dict[n])

            for n, _ in (source_names - file_names):
                self.uploadFileToItem(itemId, source_dict[n]['path'])
                ret['added'].append(source_dict[n])

        elif self.module.params['state'] == 'absent':
            for f in files:
                self.delete("file/{}".format(f['_id']))
                ret['removed'].append(f)

        if len(ret['added']) != 0 or len(ret['removed']) != 0:
            self.changed = True

        return ret

    def _get_user_by_login(self, login):
        try:
            user = self.get("/resource/lookup",
                            {"path": "/user/{}".format(login)})
        except HttpError:
            user = None
        return user

    def _get_group_by_name(self, name):
        try:
            # Could potentially fail if we have more 50 groups
            group = {g['name']: g for g in self.get("group")}['name']
        except (KeyError, HttpError):
            group = None
        return group

    def group(self, name, description, users=None, debug=False):

        r = GroupResource(self)
        valid_fields = [("name", name),
                        ("description", description)]

        if self.module.params['state'] == 'present':
            if r.name_exists(name):
                ret = r.update_by_name(name, {k: v for k, v in valid_fields
                                              if v is not None})
            else:
                ret = r.create({k: v for k, v in valid_fields
                                if v is not None})

            if users is not None:
                ret["added"] = []
                ret["removed"] = []
                ret["updated"] = []

                group_id = ret['_id']

                # Validate and normalize the user list
                for user in users:
                    assert "login" in user.keys(), \
                        "User list must have a login attribute"

                    user['type'] = self.access_types.get(
                        user.get('type', 'member'), "member")

                # dict of passed in login -> type
                user_levels = {u['login']: u['type'] for u in users}

                # dict of current login -> user information for this group
                members = {m['login']: m for m in
                           self.get('group/{}/member'.format(group_id))}

                # Add these users
                for login in (set(user_levels.keys()) - set(members.keys())):
                    user = self._get_user_by_login(login)
                    if user is not None:
                        # add user at level
                        self.post("group/{}/invitation".format(group_id),
                                  {"userId": user["_id"],
                                   "level": user_levels[login],
                                   "quiet": True,
                                   "force": True})
                        ret['added'].append(user)
                    else:
                        raise Exception('{} is not a valid login!'
                                        .format(login))

                # Remove these users
                for login in (set(members.keys()) - set(user_levels.keys())):
                    self.delete("/group/{}/member".format(group_id),
                                {"userId": members[login]['_id']})
                    ret['removed'].append(members[login])

                # Set of users that potentially need to be updated
                if len(set(members.keys()) & set(user_levels.keys())):
                    group_access = self.get('group/{}/access'.format(group_id))
                    # dict of current login -> access information for this group
                    user_access = {m['login']: m
                                   for m in group_access['access']['users']}

                    # dict of login -> level for the current group
                    # Note:
                    #  Here we join members with user_access - if the member
                    #  is not in user_access then the member has a level of 0 by
                    #  default. This gives us  a complete list of every login,
                    #  and its access level, including those that are IN the
                    #  group, but have no permissions ON the group.
                    member_levels = {m['login']:
                                     user_access.get(m['login'],
                                                     {"level": 0})['level']
                                     for m in members.values()}

                    ret = self._promote_or_demote_in_group(ret,
                                                           member_levels,
                                                           user_levels,
                                                           group_id)

                # Make sure 'changed' is handled correctly if we've
                # manipulated the group's users in any way
                if (len(ret['added']) != 0 or len(ret['removed']) != 0 or
                        len(ret['updated']) != 0):
                    self.changed = True

        elif self.module.params['state'] == 'absent':
            ret = r.delete_by_name(name)

        return ret

    def _promote_or_demote_in_group(self, ret, member_levels, user_levels,
                                    group_id):
        """Promote or demote a set of users.

        :param ret: the current dict of return values
        :param members_levels: the current access levels of each member
        :param user_levels: the desired levels of each member
        :param types: a mapping between resource names and access levels
        :returns: info about what has (or has not) been updated
        :rtype: dict

        """

        reverse_type = {v: k for k, v in self.access_types.items()}

        for login in (set(member_levels.keys()) &
                      set(user_levels.keys())):
            user = self._get_user_by_login(login)
            _id = user["_id"]

            # We're promoting
            if member_levels[login] < user_levels[login]:
                resource = reverse_type[user_levels[login]]
                self.post("group/{}/{}"
                          .format(group_id, resource),
                          {"userId": _id})

                user['from_level'] = member_levels[login]
                user['to_level'] = user_levels[login]
                ret['updated'].append(user)

            # We're demoting
            elif member_levels[login] > user_levels[login]:
                resource = reverse_type[member_levels[login]]
                self.delete("group/{}/{}"
                            .format(group_id, resource),
                            {"userId": _id})

                # In case we're not demoting to member make sure
                # to update to promote to whatever level we ARE
                # demoting too now that our user is a only a member
                if user_levels[login] != 0:
                    resource = reverse_type[user_levels[login]]
                    self.post("group/{}/{}"
                              .format(group_id, resource),
                              {"userId": _id})

                    user['from_level'] = member_levels[login]
                    user['to_level'] = user_levels[login]
                    ret['updated'].append(user)

        return ret

    def item(self, name, folderId, description=None, files=None,
             access=None, debug=False):
        ret = {}
        r = ItemResource(self, folderId)
        valid_fields = [("name", name),
                        ("description", description),
                        ('folderId', folderId)]

        if self.module.params['state'] == 'present':
            if r.name_exists(name):
                ret = r.update_by_name(name, {k: v for k, v in valid_fields
                                              if v is not None})
            else:
                ret = r.create({k: v for k, v in valid_fields
                                if v is not None})
        # handle files here

        elif self.module.params['state'] == 'absent':
            ret = r.delete_by_name(name)

        return ret

    def folder(self, name, parentId, parentType, description=None,
               public=True, folders=None, access=None, debug=False):

        ret = {}

        assert parentType in ['collection', 'folder', 'user'], \
            "parentType must be collection or folder"

        r = FolderResource(self, parentType, parentId)
        valid_fields = [("name", name),
                        ("description", description),
                        ("parentType", parentType),
                        ("parentId", parentId)]

        if self.module.params['state'] == 'present':
            if r.name_exists(name):
                ret = r.update_by_name(name, {k: v for k, v in valid_fields
                                              if v is not None})
            else:
                valid_fields = valid_fields + [("public", public)]
                ret = r.create({k: v for k, v in valid_fields
                                if v is not None})

            if folders is not None:
                self._process_folders(folders, ret["_id"], "folder")

            # handle access here
            if access is not None:
                _id = ret['_id']
                ret['access'] = self._access(r, access, _id, public=public)

        elif self.module.params['state'] == 'absent':
            ret = r.delete_by_name(name)

        return ret

    def _access(self, r, access, _id, public=True):
        access_list = {"users": [], "groups": []}
        users = access.get("users", None)
        groups = access.get("groups", None)

        if groups is not None:
            assert set(g['type'] for g in groups) <= \
                set(self.access_types.keys()), "Invalid access type!"

            # Hash of name -> group information
            # used to get user id's for access control lists
            all_groups = {g['name']: g for g in self.get("group")}

            access_list['groups'] = [{'id': all_groups[g['name']]["_id"],
                                      'level': self.access_types[g['type']]}
                                     for g in groups]

        if users is not None:

            assert set(u['type'] for u in users) <= \
                set(self.access_types.keys()), "Invalid access type!"

            # Hash of login -> user information
            # used to get user id's for access control lists
            current_users = {u['login']: self._get_user_by_login(u['login'])
                             for u in users}

            access_list['users'] = [{'id': current_users[u['login']]["_id"],
                                     "level": self.access_types[u['type']]}
                                    for u in users]

        return r.put_access(_id, access_list, public=public)

    def _process_folders(self, folders, parentId, parentType):
        """Process a list of folders from a user or collection.

        :param folders: List of folders passed as attribute
                        to user or collection
        :param parentId: ID of the user or the collection
        :param parentType: one of 'user' or 'collection'
        :returns: Nothing
        :rtype: None

        """

        current_folders = {f['name']: f for f in
                           self.get("folder", {"parentType": parentType,
                                               "parentId": parentId})}
        # Add, update or noop listed folders
        for folder in folders:
            # some validation of folder here would be a good idea
            kwargs = folder.copy()
            del kwargs['name']
            self.folder(folder['name'],
                        parentId=parentId,
                        parentType=parentType,
                        **kwargs)

        # Make sure we remove folders not listed
        for name in (set(current_folders.keys()) -
                     set([f['name'] for f in folders])):

            original_state = self.module.params['state']
            self.module.params['state'] = "absent"
            self.folder(name,
                        parentId=parentId,
                        parentType=parentType)
            self.module.params['state'] = original_state

    def collection(self, name, description=None,
                   public=True, access=None, folders=None, debug=False):

        ret = {}
        r = CollectionResource(self)
        valid_fields = [("name", name),
                        ("description", description)]

        if self.module.params['state'] == 'present':
            if r.name_exists(name):
                ret = r.update_by_name(name, {k: v for k, v in valid_fields
                                              if v is not None})
            else:
                ret = r.create({k: v for k, v in valid_fields
                                if v is not None})
        if folders is not None:
            self._process_folders(folders, ret["_id"], "collection")

        if access is not None:
            _id = ret['_id']
            ret['access'] = self._access(r, access, _id, public=public)

        elif self.module.params['state'] == 'absent':
            ret = r.delete_by_name(name)

        return ret

    def plugins(self, *plugins):
        import json
        ret = []

        available_plugins = self.get("system/plugins")
        self.message['debug']['available_plugins'] = available_plugins

        plugins = set(plugins)
        enabled_plugins = set(available_plugins['enabled'])

        # Could maybe be expanded to handle all regular expressions?
        if "*" in plugins:
            plugins = set(available_plugins['all'].keys())

        # Fail if plugins are passed in that are not available
        if not plugins <= set(available_plugins["all"].keys()):
            self.fail("%s, not available!" %
                      ",".join(list(plugins -
                                    set(available_plugins["all"].keys()))))

        # If we're trying to ensure plugins are present
        if self.module.params['state'] == 'present':
            # If plugins is not a subset of enabled plugins:
            if not plugins <= enabled_plugins:
                # Put the union of enabled_plugins nad plugins
                ret = self.put("system/plugins",
                               {"plugins":
                                json.dumps(list(plugins | enabled_plugins))})
                self.changed = True

        # If we're trying to ensure plugins are absent
        elif self.module.params['state'] == 'absent':
            # If there are plugins in the list that are enabled
            if len(enabled_plugins & plugins):

                # Put the difference of enabled_plugins and plugins
                ret = self.put("system/plugins",
                               {"plugins":
                                json.dumps(list(enabled_plugins - plugins))})
                self.changed = True

        return ret

    def user(self, login, password, firstName=None,
             lastName=None, email=None, admin=False, folders=None):

        if self.module.params['state'] == 'present':

            # Fail if we don't have firstName, lastName and email
            for var_name, var in [('firstName', firstName),
                                  ('lastName', lastName), ('email', email)]:
                if var is None:
                    self.fail("%s must be set if state "
                              "is 'present'" % var_name)

            try:
                ret = self.authenticate(username=login,
                                        password=password)

                me = self.get("user/me")

                # List of fields that can actually be updated
                updateable = ['firstName', 'lastName', 'email', 'admin']
                passed_in = [firstName, lastName, email, admin]

                # If there is actually an update to be made
                if set([(k, v) for k, v in me.items() if k in updateable]) ^ \
                   set(zip(updateable, passed_in)):

                    self.put("user/%s" % me['_id'],
                             parameters={
                                 "login": login,
                                 "firstName": firstName,
                                 "lastName": lastName,
                                 "password": password,
                                 "email": email,
                                 "admin": "true" if admin else "false"})
                    self.changed = True
            # User does not exist (with this login info)
            except AuthenticationError:
                ret = self.post("user", parameters={
                    "login": login,
                    "firstName": firstName,
                    "lastName": lastName,
                    "password": password,
                    "email": email,
                    "admin": "true" if admin else "false"
                })
                self.changed = True

            if folders is not None:
                _id = self.get("resource/lookup",
                               {"path": "/user/{}".format(login)})["_id"]
                self._process_folders(folders, _id, "user")

        elif self.module.params['state'] == 'absent':
            ret = []
            try:
                ret = self.authenticate(username=login,
                                        password=password)

                me = self.get("user/me")

                self.delete('user/%s' % me['_id'])
                self.changed = True
            # User does not exist (with this login info)
            except AuthenticationError:
                ret = []

        return ret

    # Handles patch correctly by dumping the data as a string before passing
    # it on to requests See:
    # http://docs.python-requests.org/en/master/user/quickstart/#more-complicated-post-requests
    def patch(self, path, parameters=None, data=None):
        super(GirderClientModule, self).patch(path, parameters=parameters,
                                              data=json.dumps(data))

    assetstore_types = {
        "filesystem": 0,
        "girdfs": 1,
        "s3": 2,
        "hdfs": "hdfs"
    }

    def __validate_hdfs_assetstore(self, *args, **kwargs):
        # Check if hdfs plugin is available,  enable it if it isn't
        pass

    def assetstore(self, name, type, root=None, db=None, mongohost=None,
                   replicaset='', bucket=None, prefix='', accessKeyId=None,
                   secret=None, service='s3.amazonaws.com', host=None,
                   port=None, path=None, user=None, webHdfsPort=None,
                   readOnly=False, current=False):

            # Fail if somehow we have an asset type not in assetstore_types
        if type not in self.assetstore_types.keys():
            self.fail("assetstore type %s is not implemented!" % type)

        argument_hash = {
            "filesystem": {'name': name,
                           'type': self.assetstore_types[type],
                           'root': root},
            "gridfs": {'name': name,
                       'type': self.assetstore_types[type],
                       'db': db,
                       'mongohost': mongohost,
                       'replicaset': replicaset},
            "s3": {'name': name,
                   'type': self.assetstore_types[type],
                   'bucket': bucket,
                   'prefix': prefix,
                   'accessKeyId': accessKeyId,
                   'secret': secret,
                   'service': service},
            'hdfs': {'name': name,
                     'type': self.assetstore_types[type],
                     'host': host,
                     'port': port,
                     'path': path,
                     'user': user,
                     'webHdfsPort': webHdfsPort}
        }

        # Fail if we don't have all the required attributes
        # for this asset type
        for k, v in argument_hash[type].items():
            if v is None:
                self.fail("assetstores of type "
                          "%s require attribute %s" % (type, k))

        # Set optional arguments in the hash
        argument_hash[type]['readOnly'] = readOnly
        argument_hash[type]['current'] = current

        ret = []
        # Get the current assetstores
        assetstores = {a['name']: a for a in self.get("assetstore")}

        self.message['debug']['assetstores'] = assetstores

        # If we want the assetstore to be present
        if self.module.params['state'] == 'present':

            # And the asset store exists
            if name in assetstores.keys():

                id = assetstores[name]['_id']

                ####
                # Fields that could potentially be updated
                #
                # This is necessary because there are fields in the assetstores
                # that do not hash (e.g., capacity) and fields in the
                # argument_hash that are not returned by 'GET' assetstore (e.g.
                # readOnly). We could be more precise about this
                # (e.g., by only checking items that are relevant to this type)
                # but readability suffers.
                updateable = ["root", "mongohost", "replicaset", "bucket",
                              "prefix", "db", "accessKeyId", "secret",
                              "service", "host", "port", "path", "user",
                              "webHdfsPort", "current"]

                # tuples of (key,  value) for fields that can be updated
                # in the assetstore
                assetstore_items = set((k, assetstores[name][k])
                                       for k in updateable
                                       if k in assetstores[name].keys())

                # tuples of (key,  value) for fields that can be updated
                # in the argument_hash for this assetstore type
                arg_hash_items = set((k, argument_hash[type][k])
                                     for k in updateable
                                     if k in argument_hash[type].keys())

                # if arg_hash_items not a proper subset of assetstore_items
                if not arg_hash_items <= assetstore_items:
                    # Update
                    ret = self.put("assetstore/%s" % id,
                                   parameters=argument_hash[type])

                    self.changed = True

            # And the asset store does not exist
            else:
                try:
                    # If __validate_[type]_assetstore exists then call the
                    # function with argument_hash. E.g.,  to check if the
                    # HDFS plugin is enabled
                    getattr(self, "__validate_%s_assetstore" % type
                            )(**argument_hash)
                except AttributeError:
                    pass

                ret = self.post("assetstore",
                                parameters=argument_hash[type])
                self.changed = True
        # If we want the assetstore to be gone
        elif self.module.params['state'] == 'absent':
            # And the assetstore exists
            if name in assetstores.keys():
                id = assetstores[name]['_id']
                ret = self.delete("assetstore/%s" % id,
                                  parameters=argument_hash[type])

        return ret


def main():
    """Entry point for ansible girder client module

    :returns: Nothing
    :rtype: NoneType

    """

    # Default spec for initalizing and authenticating
    argument_spec = {
        # __init__
        'host': dict(),
        'port': dict(),
        'apiRoot': dict(),
        'apiUrl': dict(),
        'scheme': dict(),
        'dryrun': dict(),
        'blacklist': dict(),

        # authenticate
        'username': dict(),
        'password': dict(),
        'token':    dict(),
        'apiKey': dict(),

        # General
        'state': dict(default="present", choices=['present', 'absent'])
    }

    gcm = GirderClientModule()

    for method in gcm.required_one_of:
        argument_spec[method] = dict(type=gcm.spec[method]['type'])

    module = AnsibleModule(  # noqa
        argument_spec=argument_spec,
        required_one_of=[gcm.required_one_of,
                         ["token", "username", "user", "apiKey"]],
        required_together=[["username", "password"]],
        mutually_exclusive=gcm.required_one_of,
        supports_check_mode=False)

    if not HAS_GIRDER_CLIENT:
        module.fail_json(msg="Could not import GirderClient!")

    try:
        gcm(module)

    except HttpError as e:
        import traceback
        module.fail_json(msg="%s:%s\n%s\n%s" % (e.__class__, str(e),
                                                e.responseText,
                                                traceback.format_exc()))
    except Exception as e:
        import traceback
        # exc_type, exc_obj, exec_tb = sys.exc_info()
        module.fail_json(msg="%s: %s\n\n%s" % (e.__class__, str(e),
                                               traceback.format_exc()))


if __name__ == '__main__':
    main()
