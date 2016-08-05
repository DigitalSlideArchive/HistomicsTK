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

# style tests
add_python_style_test(
  python_static_analysis_HistomicsTK_plugins
  "${PROJECT_SOURCE_DIR}/plugins/HistomicsTK/server"
)

add_python_style_test(
  python_static_analysis_HistomicsTK_tests
  "${PROJECT_SOURCE_DIR}/plugins/HistomicsTK/plugin_tests"
)

add_python_style_test(
  python_static_analysis_HistomicsTK_api
  "${PROJECT_SOURCE_DIR}/plugins/HistomicsTK/histomicstk"
)

# API tests
add_python_test(example PLUGIN HistomicsTK)

add_python_test(import_package PLUGIN HistomicsTK)

add_python_test(color_normalization
    SUBMODULE ReinhardNormalizationTest
    DBNAME core_color_normalization_reinhard
    PLUGIN HistomicsTK EXTERNAL_DATA
    "plugins/HistomicsTK/L1.png"    # put L1.png.sha512 in plugin_tests/data
    "plugins/HistomicsTK/Easy1.png" # put Easy1.png.sha512 in plugin_tests/data
)

add_python_test(glcm
    SUBMODULE GLCMMatrixGenerationTest
    DBNAME core_glcm_gen
    PLUGIN HistomicsTK
)

# front-end tests
add_web_client_test(
    HistomicsTK_schema "${PROJECT_SOURCE_DIR}/plugins/HistomicsTK/plugin_tests/client/parser.js"
    ENABLEDPLUGINS "HistomicsTK")

add_web_client_test(
    HistomicsTK_widget "${PROJECT_SOURCE_DIR}/plugins/HistomicsTK/plugin_tests/client/widget.js"
    ENABLEDPLUGINS "HistomicsTK")

add_web_client_test(
    HistomicsTK_visualization "${PROJECT_SOURCE_DIR}/plugins/HistomicsTK/plugin_tests/client/visualization.js"
    ENABLEDPLUGINS "HistomicsTK" "large_image")


add_eslint_test(
  js_static_analysis_HistomicsTK "${PROJECT_SOURCE_DIR}/plugins/HistomicsTK/web_client"
  ESLINT_CONFIG_FILE "${PROJECT_SOURCE_DIR}/plugins/HistomicsTK/.eslintrc"
  ESLINT_IGNORE_FILE "${PROJECT_SOURCE_DIR}/plugins/HistomicsTK/.eslintignore"
)

add_eslint_test(
  js_static_analysis_HistomicsTK_tests "${PROJECT_SOURCE_DIR}/plugins/HistomicsTK/plugin_tests/client"
  ESLINT_CONFIG_FILE "${PROJECT_SOURCE_DIR}/plugins/HistomicsTK/plugin_tests/client/.eslintrc"
  ESLINT_IGNORE_FILE "${PROJECT_SOURCE_DIR}/plugins/HistomicsTK/.eslintignore"
)
