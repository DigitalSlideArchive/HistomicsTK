find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/_compute_marginal_glcm_probs_cython.cpp
  COMMENT
    "Making ${CMAKE_CURRENT_BINARY_DIR}/_compute_marginal_glcm_probs_cython.cpp from ${CMAKE_CURRENT_SOURCE_DIR}/_compute_marginal_glcm_probs_cython.pyx"
  COMMAND Python::Interpreter -m cython --cplus -o "${CMAKE_CURRENT_BINARY_DIR}/_compute_marginal_glcm_probs_cython.cpp"
          "${CMAKE_CURRENT_SOURCE_DIR}/_compute_marginal_glcm_probs_cython.pyx"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/_compute_marginal_glcm_probs_cython.pyx"
  VERBATIM)

python_add_library(_compute_marginal_glcm_probs_cython MODULE
                   "${CMAKE_CURRENT_BINARY_DIR}/_compute_marginal_glcm_probs_cython.cpp"
                   WITH_SOABI)

target_link_libraries(_compute_marginal_glcm_probs_cython PRIVATE Python::Module Python::NumPy)

install(TARGETS _compute_marginal_glcm_probs_cython DESTINATION histomicstk/features)
