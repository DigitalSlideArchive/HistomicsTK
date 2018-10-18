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

# This gets us basic linting tests
add_standard_plugin_tests(NO_CLIENT_TESTS NO_SERVER_TESTS)

function(add_histomicstk_python_test case)
  add_python_test("${case}" PLUGIN HistomicsTK
    COVERAGE_PATHS "${PROJECT_SOURCE_DIR}/plugins/HistomicsTK/histomicstk"
    ${ARGN}
  )
endfunction()

# style tests

# We have to add the python tests and static analysis of them manually, since
# the standard plugin tests don't handle external data
add_python_style_test(
  python_static_analysis_HistomicsTK_tests
  "${PROJECT_SOURCE_DIR}/plugins/HistomicsTK/plugin_tests"
)

add_python_style_test(
  python_static_analysis_HistomicsTK_api
  "${PROJECT_SOURCE_DIR}/plugins/HistomicsTK/histomicstk"
)

# API tests
add_histomicstk_python_test(histomicstk)

add_histomicstk_python_test(image_browse_endpoints)

add_histomicstk_python_test(docker)

add_histomicstk_python_test(example)

add_histomicstk_python_test(annotation_handler)

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

add_histomicstk_python_test(global_cell_graph_features)

add_histomicstk_python_test(feature_extraction
    EXTERNAL_DATA
    # There is a bug in cmake that fails when external data files are added to
    # multiple tests, so add it in one of the tests for now
    # "plugins/HistomicsTK/L1.png"    # put L1.png.sha512 in plugin_tests/data
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
    "plugins/HistomicsTK/Easy1_nuclei_stain.npz"
    "plugins/HistomicsTK/Easy1_nuclei_fgnd_mask.npz"
    "plugins/HistomicsTK/Easy1_clog_max.npz"
    "plugins/HistomicsTK/Easy1_clog_sigma_max.npz"
    "plugins/HistomicsTK/Easy1_cdog_max.npz"
    "plugins/HistomicsTK/Easy1_cdog_sigma_max.npz"
)

add_histomicstk_python_test(cli_common
  EXTERNAL_DATA
  # Work around CMake bug when using the same image multiple times
  # "plugins/HistomicsTK/Easy1.png"
  "plugins/HistomicsTK/TCGA-06-0129-01Z-00-DX3.bae772ea-dd36-47ec-8185-761989be3cc8.svs"
  "plugins/HistomicsTK/TCGA-06-0129-01Z-00-DX3_fgnd_mask_lres.png"
  "plugins/HistomicsTK/TCGA-06-0129-01Z-00-DX3_roi_nuclei_bbox.anot"
  "plugins/HistomicsTK/TCGA-06-0129-01Z-00-DX3_roi_nuclei_boundary.anot"
)

add_histomicstk_python_test(cli_results
    # There is a bug in cmake that fails when external data files are added to
    # multiple tests, so add it in one of the tests for now
    # "plugins/HistomicsTK/Easy1.png"
    ENVIRONMENT
    "CLI_LIST_ENTRYPOINT=${PROJECT_SOURCE_DIR}/plugins/slicer_cli_web/server/cli_list_entrypoint.py"
    "CLI_CWD=${PROJECT_SOURCE_DIR}/plugins/HistomicsTK/server"
    EXTERNAL_DATA
    "plugins/HistomicsTK/TCGA-02-0010-01Z-00-DX4.07de2e55-a8fe-40ee-9e98-bcb78050b9f7-crop.tif"
)

# front-end tests

add_web_client_test(
  annotations
  "${CMAKE_CURRENT_LIST_DIR}/plugin_tests/client/annotationSpec.js"
  PLUGIN HistomicsTK
  TEST_MODULE "plugin_tests.web_client_test"
  TEST_PYTHONPATH "${CMAKE_CURRENT_LIST_DIR}"
  # EXTERNAL_DATA "plugins/HistomicsTK/sample_svs_image.TCGA-DU-6399-01A-01-TS1.e8eb65de-d63e-42db-af6f-14fefbbdf7bd.svs"
)

add_web_client_test(
  analysis
  "${CMAKE_CURRENT_LIST_DIR}/plugin_tests/client/analysisSpec.js"
  PLUGIN HistomicsTK
  TEST_MODULE "plugin_tests.web_client_test"
  TEST_PYTHONPATH "${CMAKE_CURRENT_LIST_DIR}"
  # EXTERNAL_DATA "plugins/HistomicsTK/sample_svs_image.TCGA-DU-6399-01A-01-TS1.e8eb65de-d63e-42db-af6f-14fefbbdf7bd.svs"
)

add_web_client_test(
  histomicstk
  "${CMAKE_CURRENT_LIST_DIR}/plugin_tests/client/histomicstkSpec.js"
  PLUGIN HistomicsTK
  TEST_MODULE "plugin_tests.web_client_test"
  TEST_PYTHONPATH "${CMAKE_CURRENT_LIST_DIR}"
)

add_eslint_test(
  js_static_analysis_HistomicsTK_tests "${PROJECT_SOURCE_DIR}/plugins/HistomicsTK/plugin_tests/client"
)
