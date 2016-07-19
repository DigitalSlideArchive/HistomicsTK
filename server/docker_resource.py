
#!/usr/bin/env python
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

import os
import six
import subprocess
from girder.api.v1.resource import Resource

from .constants import PluginSettings
from girder.utility.model_importer import ModelImporter


class DockerResource(Resource):
    """Manages the exposed rest api. When the settings are updated te new list
    of docker images is checked, pre-loaded images will be ignored. New images will
    cause a job to generate the cli handler and generate the rest endpoint
    asynchronously.Deleted images will result in the removal of the rest api
    endpoint though docker will still cache the image unless removed manually
    (docker rmi image_name)
    """
    loadedModules = []
