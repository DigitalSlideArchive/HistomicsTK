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

function(add_histomicstk_python_test case)
  add_python_test("${case}" PLUGIN HistomicsTK
    COVERAGE_PATHS "${PROJECT_SOURCE_DIR}/plugins/HistomicsTK/histomicstk"
    ${ARGN}
  )
endfunction()

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
add_histomicstk_python_test(docker)

add_histomicstk_python_test(example)

add_histomicstk_python_test(import_package)

add_histomicstk_python_test(color_conversion)

add_histomicstk_python_test(color_deconvolution
    SUBMODULE MacenkoTest
    DBNAME core_color_deconvolution_macenko
    EXTERNAL_DATA "plugins/HistomicsTK/Easy1.png"
)

add_histomicstk_python_test(color_deconvolution
    # Work around CMake bug when using the same image multiple times
    # EXTERNAL_DATA "plugins/HistomicsTK/Easy1.png"
)

add_histomicstk_python_test(color_normalization
    SUBMODULE ReinhardNormalizationTest
    DBNAME core_color_normalization_reinhard
    EXTERNAL_DATA
    "plugins/HistomicsTK/L1.png"    # put L1.png.sha512 in plugin_tests/data
    # Work around CMake bug when using the same image multiple times
    # "plugins/HistomicsTK/Easy1.png" # put Easy1.png.sha512 in plugin_tests/data
    "plugins/HistomicsTK/sample_svs_image.TCGA-DU-6399-01A-01-TS1.e8eb65de-d63e-42db-af6f-14fefbbdf7bd.svs"
)

add_histomicstk_python_test(color_normalization
    SUBMODULE BackgroundIntensityTest
    DBNAME core_color_normalization_background_intensity
    EXTERNAL_DATA "plugins/HistomicsTK/sample_svs_image.TCGA-DU-6399-01A-01-TS1.e8eb65de-d63e-42db-af6f-14fefbbdf7bd.svs"
)

add_histomicstk_python_test(glcm
    SUBMODULE GLCMMatrixGenerationTest
    DBNAME core_glcm_gen
)

add_histomicstk_python_test(segmentation_label
    SUBMODULE TraceBoundaryTest
    DBNAME core_trace_boundary
)


add_histomicstk_python_test(nuclei_segmentation
    SUBMODULE NucleiSegmentationTest
    DBNAME core_nuclei_seg
    EXTERNAL_DATA
    # There is a bug in cmake that fails when external data files are added to
    # multiple tests, so add it in one of the tests for now
    # "plugins/HistomicsTK/L1.png"    # put L1.png.sha512 in plugin_tests/data
    # "plugins/HistomicsTK/Easy1.png" # put Easy1.png.sha512 in plugin_tests/data
    "plugins/HistomicsTK/Easy1_nuclei_seg_kofahi_adaptive.npy" # put Easy1_nuclei_seg_kofahi_adaptive.npy.sha512 in plugin_tests/data
)

add_histomicstk_python_test(blob_detection_filters
    SUBMODULE BlobDetectionFiltersTest
    DBNAME core_blob_detection_filters
    EXTERNAL_DATA
    # There is a bug in cmake that fails when external data files are added to
    # multiple tests, so add it in one of the tests for now
    # "plugins/HistomicsTK/L1.png"
    # "plugins/HistomicsTK/Easy1.png"
    "plugins/HistomicsTK/Easy1_nuclei_stain.npy"
    "plugins/HistomicsTK/Easy1_nuclei_fgnd_mask.npy"
    "plugins/HistomicsTK/Easy1_clog_max.npy"
    "plugins/HistomicsTK/Easy1_clog_sigma_max.npy"
    "plugins/HistomicsTK/Easy1_cdog_max.npy"
    "plugins/HistomicsTK/Easy1_cdog_sigma_max.npy"
)



# front-end tests
#add_web_client_test(
#    HistomicsTK_visualization "${PROJECT_SOURCE_DIR}/plugins/HistomicsTK/plugin_tests/client/visualization.js"
#    ENABLEDPLUGINS "HistomicsTK" "large_image")
#add_web_client_test(
#    HistomicsTK_body "${PROJECT_SOURCE_DIR}/plugins/HistomicsTK/plugin_tests/client/body.js"
#    ENABLEDPLUGINS "HistomicsTK" "large_image")


add_eslint_test(
  js_static_analysis_HistomicsTK "${PROJECT_SOURCE_DIR}/plugins/HistomicsTK/web_client"
  ESLINT_CONFIG_FILE "${PROJECT_SOURCE_DIR}/plugins/HistomicsTK/.eslintrc"
  ESLINT_IGNORE_FILE "${PROJECT_SOURCE_DIR}/plugins/HistomicsTK/.eslintignore"
)

#add_eslint_test(
#  js_static_analysis_HistomicsTK_tests "${PROJECT_SOURCE_DIR}/plugins/HistomicsTK/plugin_tests/client"
#  ESLINT_CONFIG_FILE "${PROJECT_SOURCE_DIR}/plugins/HistomicsTK/plugin_tests/client/.eslintrc"
#  ESLINT_IGNORE_FILE "${PROJECT_SOURCE_DIR}/plugins/HistomicsTK/.eslintignore"
#)
